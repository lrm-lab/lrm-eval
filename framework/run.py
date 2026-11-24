#!/usr/bin/env python3
"""Run a configured benchmark and publish one record per trial."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from urllib.parse import urlsplit
from uuid import UUID

import batch_guard
import direct
import gateway_capture
import report as batch_report
import vlmevalkit
from training_capture import load_training_capture

SHARED = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().with_name("harbor.toml")


def read_json(path):
    return json.loads(path.read_text())


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            os.fchmod(handle.fileno(), 0o644)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def publication_dir(path):
    """Create directories for exported files, leaving raw jobs under the private umask."""
    if path.is_dir():
        return
    publication_dir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    else:
        path.chmod(0o755)


def publish_vlmevalkit_trajectory(trial, result, model):
    trajectory, media = vlmevalkit.atif_trajectory(result, model)
    for relative, data in media.items():
        path = trial / relative
        publication_dir(path.parent)
        atomic_write(path, data)
    atomic_write(trial / "trajectory.json", (json.dumps(trajectory, ensure_ascii=False) + "\n").encode())


def publish(job, output, metadata, training_dir=None, source_host=None, *, resumed=False, reward_key="reward"):
    """Publish only finished trial receipts; preserve valid zero rewards."""
    publication_dir(output)
    with (output / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch_status = output / "batch-status.json"
        if batch_status.is_file():
            origin = {"host": source_host or batch_guard.source_host(), "path": str(job.resolve())}
            if read_json(batch_status).get("source") != origin:
                raise ValueError(f"Batch already belongs to a different job: {output}")
        manifest = output / "manifest.json"
        if manifest.exists() and read_json(manifest) != metadata:
            raise ValueError(f"Batch already belongs to a different job: {output}")
        atomic_write(manifest, (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode())
        records = []
        identifiers = set()
        for receipt in sorted(job.glob("*/result.json")):
            try:
                result = read_json(receipt)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Skipped unreadable trial receipt: {receipt}", file=sys.stderr)
                continue
            if not result.get("finished_at"):
                continue
            trial_id = result["id"]
            if str(UUID(trial_id)) != trial_id or trial_id in identifiers:
                raise ValueError(f"Invalid or duplicate Harbor trial ID: {trial_id}")
            identifiers.add(trial_id)
            relative = Path("trials") / trial_id
            trial = output / relative
            publication_dir(trial)
            agent_info = result.get("agent_info") or {}
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            receipt_reward_key = reward_key if reward_key in rewards else "reward"
            reward = rewards.get(receipt_reward_key)
            if type(reward) not in (int, float) or not math.isfinite(reward):
                reward = None
            error = batch_guard.primary_exception(result).get("exception_type")
            diagnostic = batch_guard.failure_details(result, receipt.parent, source_host)
            status = (
                "cancelled" if diagnostic and diagnostic["kind"] == "cancelled" else
                "timeout" if error and "timeout" in error.lower() else
                "failed" if error else "scored" if reward is not None else "unscored"
            )
            if error:
                reward = None
            trajectory = receipt.parent / "agent" / "trajectory.json"
            trace_status = "missing"
            trace_path = None
            if trajectory.is_file():
                data = trajectory.read_bytes()
                try:
                    steps = json.loads(data)["steps"]
                except (ValueError, TypeError, KeyError):
                    steps = None
                if isinstance(steps, list) and all(isinstance(step, dict) for step in steps):
                    trace_status = "has_tool_calls" if any(step.get("tool_calls") for step in steps) else "no_tool_calls_recorded"
                    trace_path = str(relative / "trajectory.json")
                    atomic_write(trial / "trajectory.json", data)
                else:
                    trace_status = "invalid"
            gateway_path = None
            gateway_status = "not_captured"
            trace_file = receipt.parent / gateway_capture.TRACE_NAME
            if trace_file.is_file():
                try:
                    capture, data = gateway_capture.capture_document(trace_file, trial_id)
                except (OSError, ValueError, TypeError, AttributeError):
                    gateway_status = "invalid"
                else:
                    gateway_status = ("captured" if capture["validated"] else "unvalidated") if capture["turns"] else "empty"
                    gateway_path = str(relative / gateway_capture.TRACE_NAME)
                    atomic_write(trial / gateway_capture.TRACE_NAME, data)
            evidence = {
                "id": trial_id, "task_name": result["task_name"], "task_checksum": result.get("task_checksum"),
                "agent_info": agent_info, "verifier_rewards": rewards, "reward_key": receipt_reward_key,
                "exception_type": error,
                "failure": diagnostic,
                "started_at": result.get("started_at"), "finished_at": result["finished_at"],
            }
            usage = episode_usage(result)
            if usage:
                evidence["usage"] = usage
            if "direct_result" in result:
                evidence["direct_result"] = result["direct_result"]
                trace_status = "direct_response"
            if "vlmeval_result" in result:
                vlm = result["vlmeval_result"]
                evidence["vlmeval_result"] = {
                    key: value for key, value in vlm.items() if key not in ("request_messages", "prediction")
                }
                http_status = vlm.get("http_status")
                if error and isinstance(http_status, int) and 200 <= http_status < 300:
                    diagnostic.update(kind="model_empty_answer", summary="模型网关响应成功，缺少可评分的回答。")
                elif error and isinstance(http_status, int):
                    diagnostic.update(kind="model_gateway_http_error", summary=f"模型网关返回 HTTP {http_status}。")
                elif not error and vlm.get("request_messages") and vlm.get("prediction"):
                    try:
                        publish_vlmevalkit_trajectory(trial, result, metadata["model"])
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"Could not convert VLMEvalKit trace {trial_id}: {type(exc).__name__}",
                              file=sys.stderr)
                        trace_status = "invalid"
                    else:
                        trace_status = "no_tool_calls_recorded"
                        trace_path = str(relative / "trajectory.json")
            atomic_write(trial / "result.json", (json.dumps(evidence, ensure_ascii=False) + "\n").encode())
            training_path = None
            training_status = "not_captured"
            checkpoint = None
            if training_dir and (training_dir / trial_id).exists():
                capture, trace_bytes = load_training_capture(training_dir, trial_id)
                checkpoint = capture["checkpoint"]
                training_path = str(relative / "training_trace.json")
                training_status = "validated" if status == "scored" and reward is not None else "unscored"
                atomic_write(trial / "training_trace.json", trace_bytes)
                atomic_write(trial / "capture.json", (json.dumps(capture, ensure_ascii=False) + "\n").encode())
            records.append({
                "schema_version": 1, "trial_id": trial_id, "dataset": metadata["dataset"],
                "task_id": result["task_name"], "task_checksum": result.get("task_checksum"),
                "harness": metadata["harness"], "harness_version": agent_info.get("version"),
                "runner": metadata.get("runner", "harbor"),
                "model_route": metadata["model"], "checkpoint": checkpoint,
                "status": status, "reward": reward, "verifier_rewards": rewards, "exception_type": error,
                "failure": diagnostic,
                "usage": usage,
                "result": str(relative / "result.json"), "trajectory": trace_path,
                "trajectory_status": trace_status, "training_trace": training_path, "training_status": training_status,
                "gateway_trace": gateway_path, "gateway_status": gateway_status,
            })
        index = output / "episodes.jsonl"
        if records or (resumed and index.is_file()):
            merged = {}
            if index.is_file():
                for line in index.read_text().splitlines():
                    previous = json.loads(line)
                    if resumed and previous.get("exception_type") == "CancelledError" and previous["trial_id"] not in identifiers:
                        continue
                    merged[previous["trial_id"]] = previous
            for row in records:
                previous = merged.get(row["trial_id"])
                if previous and previous["training_trace"] and not row["training_trace"]:
                    for key in ("training_trace", "training_status", "checkpoint"):
                        row[key] = previous[key]
                merged[row["trial_id"]] = row
            atomic_write(index, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                         for row in merged.values()).encode())
        return len(records)


def episode_usage(result):
    metrics = {}
    contexts = [result["agent_result"]] if result.get("agent_result") else [
        step["agent_result"] for step in result.get("step_results") or [] if step.get("agent_result")
    ]
    sources = {"input_tokens": "n_input_tokens", "output_tokens": "n_output_tokens", "cost_usd": "cost_usd"}
    for name, source in sources.items():
        values = [item[source] for item in contexts if batch_report.finite(item.get(source))]
        if values:
            metrics[name] = sum(values)
    direct_usage = (result.get("direct_result") or {}).get("usage") or {}
    for name, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        if batch_report.finite(direct_usage.get(source)):
            metrics[name] = direct_usage[source]
    try:
        start = datetime.fromisoformat(result["started_at"])
        end = datetime.fromisoformat(result["finished_at"])
        seconds = (end - start).total_seconds()
        if seconds >= 0:
            metrics["duration_seconds"] = seconds
    except (TypeError, ValueError, KeyError):
        pass
    return metrics


def write_batch_report(output, planned=None):
    try:
        _, rows, metadata, status = batch_report.read_batch(output)
        official = output / "official-result.json"
        result = batch_report.summarize(rows, metadata, status, planned=planned,
                                        official=official if official.is_file() else None)
        batch_report.write_report(output / "batch-report.json",
                                  json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Could not write batch report: {error}", file=sys.stderr)


def command_for(args, dataset, bundle, suite):
    """Keep installed handoff profiles compatible and support native Harbor datasets."""
    if "handoff_profile" in suite and args.agent in ("pi", "openclaw", "claude-code"):
        cmd = handoff_command(args, dataset, bundle, suite["handoff_profile"])
    else:
        cmd = native_harbor_command(args, dataset, suite)
    if args.api_format:
        cmd.extend(["--api-format", args.api_format])
    if args.mounts:
        cmd.extend(["--mounts", json.dumps(args.mounts)])
    for value in args.agent_kwarg:
        cmd.extend(["--ak", value])
    for value in args.agent_env:
        cmd.extend(["--agent-env", value])
    for value in args.verifier_env:
        cmd.extend(["--verifier-env", value])
    return cmd


def native_harbor_command(args, dataset, suite):
    selected = list(dict.fromkeys(args.task))
    dataset = dataset.resolve()
    for name in selected:
        task = dataset / name / "task.toml"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or not task.is_file() or not task.resolve().is_relative_to(dataset):
            raise ValueError(f"Task is missing from dataset: {name}")
    runner = os.getenv("PI_BENCH_RUNNER_BIN")
    if runner:
        executable = Path(runner).resolve() / "harbor"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Bundled runner is missing or not executable: {executable}")
        cmd = [str(executable)]
    else:
        cmd = ["uvx", "--from", "harbor==0.26.0", "--with", "litellm==1.110.1", "harbor"]
    path = dataset / selected[0] if len(selected) == 1 else dataset
    agent = suite.get("agents", {}).get(args.agent, args.agent)
    cmd.extend(["run", "-p", str(path), "--agent", agent,
                "--env", "docker", "--jobs-dir", str(args.work / "jobs"), "--job-name", args.batch,
                "--n-concurrent", str(args.concurrent), "--n-attempts", str(args.attempts)])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.agent_version:
        cmd.extend(["--ak", f"version={args.agent_version}"])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
        if args.agent == "pi" and not args.api_format:
            cmd.extend(["--ak", "model_api=openai-completions"])
    if args.thinking and args.agent in ("pi", "openclaw", "claude-code"):
        cmd.extend(["--ak", f"reasoning_effort={args.thinking}"])
    if len(selected) > 1:
        for task in selected:
            cmd.extend(["--include-task-name", task])
    return cmd


def handoff_command(args, dataset, bundle, profile):
    """Keep the existing Terminal-Bench and WildClaw agent settings."""
    sys.path.insert(0, str(bundle / "code"))
    from benchpi.__main__ import cli_args
    from types import SimpleNamespace

    template = SimpleNamespace(
        dataset=dataset, suite=profile, task=args.task, agent=args.agent if args.agent != "claude-code" else "pi",
        model=args.model, thinking=args.thinking, version=args.agent_version, env="docker", jobs=args.work / "jobs",
        job_name=args.batch, concurrent=args.concurrent, attempts=args.attempts, env_file=None,
        base_url=args.base_url, setup_timeout_multiplier=10,
    )
    cmd = cli_args(template)
    if args.agent == "claude-code":
        cmd[cmd.index("--agent") + 1] = "claude-code"
        if "model_api=openai-completions" in cmd:
            index = cmd.index("model_api=openai-completions")
            del cmd[index - 1:index + 1]
    return cmd


def prepare_wildclaw(source, bundle, work, suite_id):
    working_root = work / "prepared" / suite_id
    destination = working_root / "datasets" / "wildclaw-run"
    working_root.mkdir(parents=True, exist_ok=True)
    with (working_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=working_root) as temp:
                staging = Path(temp)
                (staging / "datasets").mkdir()
                (staging / "install").mkdir()
                script = staging / "install" / "prepare-wildclaw.py"
                shutil.copyfile(bundle / "install" / script.name, script)
                shutil.copytree(source,
                                staging / "datasets" / "wildclaw-harbor", symlinks=True)
                subprocess.run([sys.executable, str(script)], check=True)
                destination.parent.mkdir(exist_ok=True)
                (staging / "datasets" / "wildclaw-run").rename(destination)
    return destination


def run_lock(args, cmd, dataset, output, cwd, env):
    prefix = cmd[:cmd.index("run")]
    runner_version = subprocess.check_output(
        [*prefix, "--version"], cwd=cwd, env=env, text=True, timeout=60
    ).strip()
    hardware_files = {}
    for mount in args.mounts:
        if mount.get("type") != "bind" or not mount.get("source"):
            continue
        path = Path(mount["source"]).resolve()
        if path.is_file():
            hardware_files[str(path)] = batch_guard.file_sha256(path)
        elif path.is_dir():
            hardware_files[str(path)] = batch_guard.document_sha256(list(batch_guard.tree_entries(path)))
    version = next((cmd[i + 1].partition("=")[2] for i in range(len(cmd) - 1, 0, -1)
                    if i + 1 < len(cmd) and cmd[i] in ("--ak", "--agent-kwarg")
                    and cmd[i + 1].startswith("version=")), None)
    return {
        "schema_version": 1, "dataset": batch_guard.dataset_sha256(dataset, args.task),
        "model": args.model, "harness": args.agent, "harness_version": version,
        "attempts": args.attempts, "concurrent": args.concurrent, "harbor_version": runner_version,
        "invocation_sha256": batch_guard.document_sha256([str(cwd), cmd]),
        "mounts_sha256": batch_guard.document_sha256(hardware_files),
        "output": str(output.resolve()),
        **({"reward_key": args.reward_key} if args.reward_key != "reward" else {}),
    }


def execute_harbor(args, cmd, dataset, output, metadata, cwd, env):
    job = args.work / "jobs" / args.batch
    origin = {"host": args.source_host, "path": str(job)}
    lock_dir = args.work / "batch-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{args.batch}.lease").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This batch already has an active Tianxi runner") from None
        saved = lock_dir / f"{args.batch}.json"
        if args.resume and not saved.is_file():
            raise ValueError("No Tianxi resume lock for this batch")
        if not args.resume and (job.exists() or output.exists()):
            raise ValueError("Batch already exists. Use --resume, --export-job, or a fresh name")

        expected = run_lock(args, cmd, dataset, output, cwd, env)
        tasks = [dataset / task / "task.toml" for task in dict.fromkeys(args.task)] if args.task else sorted(
            dataset.glob("*/task.toml")
        )
        task_attempts = Counter()
        for task in tasks:
            definition = tomllib.loads(task.read_text()).get("task") or {}
            task_attempts[definition.get("name") or task.parent.name] += args.attempts
        planned = sum(task_attempts.values())
        pause_path = job / batch_guard.PAUSE_FILE
        status_path = output / "batch-status.json"
        if args.resume:
            old = read_json(saved)
            changed = sorted(key for key in old.keys() | expected.keys() if old.get(key) != expected.get(key))
            if changed:
                raise ValueError(f"Resume lock mismatch in: {', '.join(changed)}. Use a new batch for changed inputs.")
            if (args.agent not in ("oracle", "nop")
                    and expected["harness_version"] in (None, "", "latest", "stable", "main", "master")):
                raise ValueError("The original batch needs a pinned harness version for checked resume")
            if not (job / batch_guard.INTEGRITY_FILE).is_file():
                raise ValueError("The original job did not reach its first trial; native integrity is unavailable")
            if read_json(job / batch_guard.INTEGRITY_FILE) != batch_guard.native_integrity(job):
                raise ValueError("Native Harbor config.json or lock.json changed since dispatch")
            if (output / "manifest.json").is_file() and read_json(output / "manifest.json") != metadata:
                raise ValueError("The shared batch manifest belongs to a different run")
            if status_path.is_file() and read_json(status_path).get("source") != origin:
                raise ValueError("The shared batch belongs to a different execution host or job")
            pause_path.unlink(missing_ok=True)
            cmd = [*cmd[:cmd.index("run")], "jobs", "resume", "--job-path", str(job)]
            publication_dir(output)
        else:
            publication_dir(output.parent)
            try:
                output.mkdir()
            except FileExistsError:
                raise ValueError(f"The shared batch name is already reserved: {output}") from None
            output.chmod(0o755)
            batch_guard.write_private_json(saved, expected)
        running = {"status": "running", "source": origin, "planned": planned, "task_attempts": task_attempts}
        atomic_write(status_path, (json.dumps(running) + "\n").encode())

        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(DEFAULT_CONFIG.parent), env.get("PYTHONPATH")
        )))
        cmd.extend(["--plugin", batch_guard.PLUGIN,
                    "--pk", f"consecutive={args.pause_after}", "--pk", f"host={args.source_host}"])
        try:
            returncode = subprocess.run(cmd, cwd=cwd, env=env).returncode
        except KeyboardInterrupt:
            returncode = 130
        if returncode < 0:
            returncode = 128 - returncode

        count = publish(job, output, metadata, args.training_dir, args.source_host,
                        resumed=args.resume, reward_key=args.reward_key) if job.is_dir() else 0
        print(f"Published {count} trial(s) to {output}")
        pause = read_json(pause_path) if pause_path.is_file() else None
        progress_file = job / "result.json"
        progress = read_json(progress_file) if progress_file.is_file() else {}
        stats = progress.get("stats") or {}
        state = {
            "status": "paused" if pause else "interrupted" if returncode in (130, 143) else
                      "failed" if returncode or not count else "finished",
            "completed": stats.get("n_completed_trials"), "planned": planned, "task_attempts": task_attempts,
            "harbor_planned": progress.get("n_total_trials"),
            "consecutive_failure": pause, "source": origin,
        }
        atomic_write(status_path, (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode())
        write_batch_report(output)
        if pause:
            print(f"Dispatch paused after {pause['consecutive']} {pause['kind']} failures. "
                  "Use --resume with the original inputs after resolving the cause.", file=sys.stderr)
            return batch_guard.PAUSED_EXIT
        return returncode or (0 if count else 1)



def execute_gateway(args, dataset, output, metadata):
    _, separator, model = args.model.partition("/")
    if not separator:
        model = args.model
    tasks = ([dataset / name for name in dict.fromkeys(args.task)] if args.task
             else sorted(path.parent for path in dataset.glob("*/task.toml")))
    if not tasks:
        raise ValueError(f"No Harbor tasks found in {dataset}")
    for task in tasks:
        if not (task / "task.toml").is_file() or not task.resolve().is_relative_to(dataset):
            raise ValueError(f"Task is missing from dataset: {task.name}")
    if args.plan:
        print(json.dumps({"runner": "openenv-gateway", "tasks": [p.name for p in tasks],
                          "model": model, "harness": args.agent, "attempts": args.attempts,
                          "concurrent": args.concurrent, "capture_host": args.capture_host}))
        return 0
    if not args.base_url:
        raise ValueError("Set OPENAI_BASE_URL or --base-url for gateway capture")
    key = os.getenv("OPENAI_API_KEY")
    job = args.work / "jobs" / args.batch
    if output.exists() or job.exists():
        raise ValueError("Gateway batch already exists. Use a fresh name")
    task_attempts = Counter()
    for task in tasks:
        definition = tomllib.loads((task / "task.toml").read_text()).get("task") or {}
        task_attempts[definition.get("name") or task.name] += args.attempts
    planned = sum(task_attempts.values())
    job.mkdir(parents=True, exist_ok=False)
    publication_dir(output.parent)
    output.mkdir()
    output.chmod(0o755)
    state = {"status": "running", "source": {"host": args.source_host, "path": str(job)},
             "planned": planned, "task_attempts": task_attempts}

    def write_document(path, document):
        atomic_write(path, (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode())

    status = output / "batch-status.json"
    write_document(status, state)
    counts = Counter()
    try:
        counts = gateway_capture.run_capture(args, tasks, job, args.base_url, model, key, write_document)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    finally:
        count = publish(job, output, metadata, source_host=args.source_host, reward_key=args.reward_key)
        success = count == planned and counts["captured"] == planned
        if state["status"] == "running":
            state["status"] = "paused" if counts["paused"] else "finished" if success else "failed"
        state["completed"] = count
        write_document(status, state)
        write_batch_report(output)
    print(f"Published {count} trial(s) to {output}")
    return batch_guard.PAUSED_EXIT if counts["paused"] else int(not success)


def publish_vlmevalkit_metrics(job, output):
    for relative in ("vlmevalkit/metrics.json", "official-result.json"):
        source = job / relative
        if source.is_file():
            target = output / relative
            publication_dir(target.parent)
            atomic_write(target, source.read_bytes())


def execute_vlmevalkit(args, suite, dataset, output, metadata):
    options = vlmevalkit.options_for(suite, dataset)
    if (args.attempts != 1 or args.api_format or args.agent_version or args.agent_kwarg
            or args.agent_env or args.verifier_env or args.mounts or args.training_dir):
        raise ValueError("VLMEvalKit uses Chat Completions with one prediction per sample")
    mode = "infer" if args.task else options.get("mode", "all")
    if args.plan:
        print(json.dumps({"runner": "vlmevalkit", "dataset": options["dataset"], "path": str(dataset),
                          "tasks": args.task or "all", "model": args.model, "mode": mode}, ensure_ascii=False))
        return 0
    if not args.base_url or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("VLMEvalKit requires OPENAI_BASE_URL and OPENAI_API_KEY")
    job = args.work / "jobs" / args.batch
    if job.exists() or output.exists():
        raise ValueError("Batch already exists. Use a fresh name or --export-job")
    publication_dir(output.parent)
    output.mkdir()
    output.chmod(0o755)
    status = output / "batch-status.json"
    state = {"source": {"host": args.source_host, "path": str(job)}, "status": "running"}
    atomic_write(status, (json.dumps(state) + "\n").encode())
    try:
        exit_code = vlmevalkit.launch(args, suite, dataset, job, mode)
        state["status"] = "finished" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    finally:
        if state["status"] == "running":
            state["status"] = "failed"
        count = publish(job, output, metadata, source_host=args.source_host)
        publish_vlmevalkit_metrics(job, output)
        state["completed"] = count
        if (job / "result.json").is_file():
            state["planned"] = read_json(job / "result.json")["n_total_trials"]
        atomic_write(status, (json.dumps(state) + "\n").encode())
        write_batch_report(output)
        print(f"Published {count} trial(s) to {output}")
    return exit_code


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", help="Dataset ID in the [datasets] table")
    parser.add_argument("--agent", help="Harbor agent name or import path")
    parser.add_argument("--agent-kwarg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--agent-env", action="append", default=[], metavar="KEY=VALUE",
                        help="Harbor agent environment assignment. Reference secrets as ${VAR}.")
    parser.add_argument("--verifier-env", action="append", default=[], metavar="KEY=VALUE",
                        help="Harbor verifier environment assignment. Reference secrets as ${VAR}.")
    parser.add_argument("--agent-version")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--task", action="append", default=[])
    tasks.add_argument("--all", action="store_true", help="Run all tasks in a Harbor or VLMEvalKit dataset")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-format", help="Harbor model API format for the configured gateway")
    parser.add_argument("--mounts", type=json.loads, default=[], help="Harbor Docker bind mounts as a JSON array")
    parser.add_argument("--agent-ca-bundle", type=Path, help="Trusted CA bundle to mount into the agent")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--pause-after", type=int, default=3,
                        help="Pause Harbor dispatch after this many consecutive matching run errors; 0 disables")
    parser.add_argument("--source-host", default=batch_guard.source_host(),
                        help="Execution host in published raw log locators")
    parser.add_argument("--work", type=Path, default=Path.home() / ".cache" / "tianxi-harbor")
    parser.add_argument("--output", type=Path, default=SHARED / "harbor-runs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path, help="Local copy of the handoff bundle")
    parser.add_argument("--training-dir", type=Path, help="Trial-bound captures produced by the OpenEnv collector")
    parser.add_argument("--gateway-capture", action="store_true",
                        help="Run Harbor via the OpenEnv proxy and save actual model request windows")
    parser.add_argument("--capture-host", default="172.17.0.1",
                        help="Docker-reachable local IPv4 address on which the capture gateway listens")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--export-job", type=Path, help="Import completed trial receipts from a local job")
    mode.add_argument("--resume", action="store_true", help="Continue a locked local Harbor batch")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.batch):
        parser.error("Invalid batch name")
    if args.concurrent < 1 or args.attempts < 1 or args.pause_after < 0:
        parser.error("Concurrency and attempts must be positive, and pause-after must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9.:-]+", args.source_host):
        parser.error("Invalid source host")
    config = tomllib.loads(args.config.read_text())
    datasets = config.get("datasets", {})
    if args.suite not in datasets or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.suite):
        parser.error(f"Unknown dataset ID {args.suite!r}. Configured: {', '.join(sorted(datasets))}")
    suite = datasets[args.suite]
    args.reward_key = suite.get("reward_key", "reward")
    if not isinstance(args.reward_key, str) or not args.reward_key.strip():
        parser.error("Dataset reward_key must be a nonempty string")
    config_dir = args.config.resolve().parent
    bundle_path = args.bundle or os.getenv("TIANXI_BENCH_BUNDLE") or config.get("bundle")
    bundle = (config_dir / Path(bundle_path).expanduser()).resolve() if bundle_path else None
    path_root = suite.get("path_root", "bundle" if "handoff_profile" in suite else "config")
    if path_root not in ("bundle", "config"):
        parser.error("Dataset path_root must be bundle or config")
    needs_bundle = path_root == "bundle" or "handoff_profile" in suite or suite.get("prepare")
    if needs_bundle and bundle is None:
        parser.error("The selected dataset setup requires a handoff bundle")
    if suite.get("prepare") not in (None, "wildclaw-crlf"):
        parser.error("Unknown dataset preparation method")
    runner = suite.get("runner", "harbor")
    if runner not in ("harbor", "direct", "vlmevalkit"):
        parser.error("Dataset runner must be harbor, direct, or vlmevalkit")
    if args.resume and runner != "harbor":
        parser.error("Checked resume is available for Harbor datasets")
    if runner != "harbor" and args.agent is not None:
        parser.error(f"{runner} datasets are run by the model API. Omit --agent.")
    args.agent = runner if runner != "harbor" else args.agent or "pi"
    if not args.model and args.agent not in ("oracle", "nop"):
        parser.error("Specify --model for the selected harness")
    metadata = {"schema_version": 1, "dataset": suite["name"], "harness": args.agent,
                "model": args.model, "batch": args.batch}
    if runner != "harbor":
        metadata["runner"] = runner
    output = args.output / args.batch
    if args.export_job:
        count = publish(args.export_job.resolve(), output, metadata, args.training_dir, args.source_host,
                        reward_key=args.reward_key)
        if count == 0:
            parser.error("No finished trial receipts found")
        if runner == "vlmevalkit":
            publish_vlmevalkit_metrics(args.export_job, output)
        original_result = args.export_job / "result.json"
        try:
            planned = read_json(original_result).get("n_total_trials") if original_result.is_file() else None
        except (OSError, ValueError):
            planned = None
        write_batch_report(output, planned=planned if not (output / "batch-status.json").is_file() else None)
        print(f"Published {count} trial(s) to {output}")
        return 0

    if not args.task and not args.all:
        parser.error("Specify at least one --task or use --all")
    if args.all and runner == "direct":
        parser.error("Select direct cases with --task")
    if runner == "direct" and args.api_format:
        parser.error("Direct runs use OpenAI Chat Completions")
    if runner == "direct" and args.verifier_env:
        parser.error("--verifier-env requires a Harbor dataset")
    if not isinstance(args.mounts, list) or not all(isinstance(mount, dict) for mount in args.mounts):
        parser.error("--mounts requires a JSON array of mount objects")
    if args.agent_ca_bundle:
        ca_bundle = args.agent_ca_bundle.resolve()
        if runner != "harbor" or not ca_bundle.is_file():
            parser.error("--agent-ca-bundle requires Harbor and a readable CA file")
        target = "/etc/ssl/certs/tianxi-host-ca.pem"
        if any(mount.get("target") == target for mount in args.mounts):
            parser.error(f"--agent-ca-bundle already supplies the mount at {target}")
        args.mounts.append({"type": "bind", "source": str(ca_bundle), "target": target, "read_only": True})
        args.agent_env.extend([f"SSL_CERT_FILE={target}", f"NODE_EXTRA_CA_CERTS={target}"])
    dataset_root = bundle if path_root == "bundle" else config_dir
    dataset = (dataset_root / suite["path"]).resolve()
    args.work = args.work.resolve()
    if bundle is not None:
        bundled_bin = bundle / "install" / "runner-env" / "bin"
        if os.access(bundled_bin / "harbor", os.X_OK):
            os.environ.setdefault("PI_BENCH_RUNNER_BIN", str(bundled_bin))
    if not args.base_url and (args.gateway_capture or runner != "harbor" or args.agent in ("pi", "openclaw", "claude-code")
                              or args.api_format in ("openai_chat_completions", "openai_responses")):
        variable = "ANTHROPIC_BASE_URL" if args.agent == "claude-code" and not args.gateway_capture else "OPENAI_BASE_URL"
        args.base_url = os.getenv(variable)
    if args.agent == "claude-code":
        args.agent_version = args.agent_version or config.get("claude_code_version")
        if not args.agent_version:
            parser.error("Specify a pinned --agent-version for Claude Code")
    if args.agent == "opencode":
        args.agent_version = args.agent_version or config.get("opencode_version")
        target = "/usr/local/bin/opencode"
        custom_binary = any(mount.get("target") == target for mount in args.mounts)
        if not args.gateway_capture and bundle and not custom_binary and args.agent_version and re.fullmatch(r"\d+\.\d+\.\d+", args.agent_version):
            binary = bundle / "runtime" / f"opencode-{args.agent_version}" / "opencode"
            if os.access(binary, os.X_OK):
                args.mounts.append({"type": "bind", "source": str(binary), "target": target, "read_only": True})
                if not any(value.startswith("skip_install=") for value in args.agent_kwarg):
                    args.agent_kwarg.append("skip_install=true")
    if args.base_url:
        url = urlsplit(args.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error("Use an HTTP(S) gateway URL without credentials, query parameters, or fragments")
    if args.gateway_capture:
        if (runner != "harbor" or args.resume or args.agent in ("oracle", "nop")
                or args.agent_env or args.agent_kwarg or args.verifier_env or args.mounts
                or args.api_format or args.agent_ca_bundle or args.training_dir):
            parser.error("Gateway capture uses OpenEnv harness settings. It requires a new Harbor model batch "
                         "and does not take native agent, verifier, or mount overrides")
        try:
            if suite.get("prepare") == "wildclaw-crlf" and not args.plan:
                dataset = prepare_wildclaw(dataset, bundle, args.work, args.suite)
            return execute_gateway(args, dataset, output, metadata)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))

    if runner == "vlmevalkit":
        try:
            return execute_vlmevalkit(args, suite, dataset, output, metadata)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(str(error))

    if runner == "direct":
        try:
            cases = direct.load_tasks(dataset, args.task)
            scorer = (config_dir / suite["scorer"]).resolve() if suite.get("scorer") else None
            if scorer and not scorer.is_file():
                raise ValueError(f"Missing direct scorer: {scorer}")
            for case in cases:
                direct.model_messages(case, dataset.parent)
                if scorer is None and "answer" not in case:
                    raise ValueError(f"Missing exact-match answer for {case['id']}")
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        if args.plan:
            print(json.dumps({"runner": "direct", "dataset": str(dataset), "tasks": args.task,
                              "model": args.model, "scorer": str(scorer) if scorer else "exact_match"}))
            return 0
        if not args.base_url or not os.getenv("OPENAI_API_KEY"):
            parser.error("Direct runs require OPENAI_BASE_URL and OPENAI_API_KEY")
        job = args.work / "jobs" / args.batch
        if job.exists() or output.exists():
            parser.error("Batch already exists. Use a fresh name or --export-job")
        success = direct.run(cases, dataset, job, args, suite, scorer, os.environ["OPENAI_API_KEY"])
        count = publish(job, output, metadata, args.training_dir, args.source_host)
        write_batch_report(output, planned=len(cases))
        print(f"Published {count} trial(s) to {output}")
        return 0 if success else 1

    if needs_bundle and not bundle.is_dir():
        parser.error(f"Handoff bundle missing at {bundle}. Run setup.sh or set --bundle.")
    if not any(dataset.glob("*/task.toml")):
        parser.error(f"No Harbor tasks found in {dataset}")
    if args.plan:
        try:
            print(shlex.join(command_for(args, dataset, bundle, suite)))
        except (ImportError, OSError, ValueError) as error:
            parser.error(str(error))
        return 0

    if not shutil.which("docker"):
        parser.error("Docker is required on the execution host")
    if args.agent in ("pi", "openclaw", "claude-code") and not args.base_url:
        parser.error("Set the gateway URL for the selected agent protocol")
    if args.agent == "claude-code":
        if not (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")):
            parser.error("Set Claude Code gateway credentials in the environment")
    elif args.agent in ("pi", "openclaw") and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the environment")
    if not os.getenv("PI_BENCH_RUNNER_BIN") and not shutil.which("uvx"):
        parser.error("Set PI_BENCH_RUNNER_BIN to the unpacked runner bin or install uvx")
    if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL).returncode:
        parser.error("The execution account cannot reach Docker")
    if suite.get("prepare") == "wildclaw-crlf":
        dataset = prepare_wildclaw(dataset, bundle, args.work, args.suite)
    try:
        cmd = command_for(args, dataset, bundle, suite)
    except (ImportError, OSError, ValueError) as error:
        parser.error(str(error))
    env = dict(os.environ)
    cwd = config_dir
    if bundle is not None and (bundle / "code").is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(bundle / "code"), env.get("PYTHONPATH"))))
        env["PI_BENCH_INSTALL_CACHE"] = str(bundle / "runtime" / "harbor-cache")
        env["PI_BENCH_AGENT_FILES"] = str(bundle / "code" / "agent-profiles" / "v5")
        env["PI_BENCH_RUNTIME"] = "1"
        cwd = bundle / "code"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        return execute_harbor(args, cmd, dataset, output, metadata, cwd, env)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

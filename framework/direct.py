"""Run model-only tasks against a Chat Completions endpoint, without Docker."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def load_tasks(path, selected):
    cases = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        task_id = case["id"]
        if not isinstance(task_id, str) or not task_id or task_id in cases:
            raise ValueError(f"Missing or duplicate task ID: {task_id!r}")
        cases[task_id] = case
    missing = set(selected) - cases.keys()
    if missing:
        raise ValueError(f"Missing task IDs: {', '.join(sorted(missing))}")
    return [cases[name] for name in dict.fromkeys(selected)]


def model_messages(case, root):
    prompt = case["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing prompt for {case['id']}")
    images = case.get("images", [])
    if not isinstance(images, list):
        raise ValueError("images must be a list of relative file paths")
    if not images:
        return [{"role": "user", "content": prompt}]

    content = [{"type": "text", "text": prompt}]
    for name in images:
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Image path must be relative to the dataset folder")
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError(f"Unsupported image type: {name}")
        image = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})
    return [{"role": "user", "content": content}]


def score(scorer, case, prediction, timeout):
    if scorer is None:
        return float(prediction.strip() == str(case["answer"]).strip())
    result = subprocess.run(
        [sys.executable, str(scorer)],
        input=json.dumps({"case": case, "prediction": prediction}),
        text=True, capture_output=True, check=True, cwd=scorer.parent, timeout=timeout,
    )
    reward = json.loads(result.stdout)["reward"]
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("Scorer must return a finite numeric reward")
    return reward


def run(cases, dataset, job, args, suite, scorer, key):
    job.mkdir(parents=True)
    timeout = suite.get("timeout_sec", 120)
    model = args.model
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def trial(case):
        trial_id = str(uuid4())
        folder = job / trial_id
        folder.mkdir()
        started = datetime.now(timezone.utc).isoformat()
        body = {"model": model, "messages": model_messages(case, dataset.parent),
                "stream": False, "temperature": suite.get("temperature", 0)}
        if "max_tokens" in suite:
            body["max_tokens"] = suite["max_tokens"]
        request = Request(endpoint, data=json.dumps(body).encode(), method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        error = None
        reward = None
        prediction = None
        actual_model = None
        usage = None
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            prediction = payload["choices"][0]["message"]["content"]
            if not isinstance(prediction, str):
                raise ValueError("Expected a text completion")
            actual_model = payload.get("model")
            usage = payload.get("usage")
            reward = score(scorer, case, prediction, timeout)
        except (HTTPError, URLError, TimeoutError, ValueError, IndexError, KeyError,
                subprocess.SubprocessError) as exc:
            error = ("TimeoutError" if isinstance(getattr(exc, "reason", exc), TimeoutError)
                     else f"HTTPError{exc.code}" if isinstance(exc, HTTPError)
                     else type(exc).__name__)
        result = {
            "id": trial_id, "task_name": case["id"],
            "task_checksum": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "agent_info": {"name": "direct", "model_info": {"name": actual_model}},
            "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
            "exception_info": {"exception_type": error} if error else None,
            "direct_result": {"prediction": prediction, "reported_model": actual_model, "usage": usage},
        }
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return error is None

    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        return all(list(pool.map(trial, (case for case in cases for _ in range(args.attempts)))))

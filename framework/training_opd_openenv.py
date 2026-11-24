"""OpenEnv Harbor rollout adapter and checkpoint publishing for Tianxi OPD."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

from training_opd import Rollout, TokenTurn


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    api_key: str = ""


def captured_turns(result, temperature):
    from openenv.core.harness.capture.upstream import truncating_params
    from openenv.harbor.contract import to_training_trace

    if not result.ok or result.rollout_type != "train" or result.reward is None:
        raise ValueError(f"OpenEnv rollout is unusable for OPD: {result.error or result.rollout_type}")
    if result.param_fixes:
        raise ValueError("OpenEnv rollout changed the requested policy")
    turns = []
    for turn in to_training_trace(result).turns:
        if not any(turn.loss_mask):
            continue
        sampling = turn.metadata["sampling_params"]
        if sampling.get("temperature") != temperature or truncating_params(sampling):
            raise ValueError("OpenEnv did not record the requested full-vocabulary sampling policy")
        turns.append(TokenTurn(tuple(turn.prompt_token_ids), tuple(turn.completion_token_ids),
                               tuple(turn.per_token_logps), tuple(turn.loss_mask[len(turn.prompt_token_ids):])))
    if not turns:
        raise ValueError("OpenEnv rollout has no supervised tokens")
    return tuple(turns)


class OpenEnvRollout:
    def __init__(self, args, settings):
        del args
        from openenv.harbor.client import HarborEnv

        self.options = settings.get("openenv", {})
        self.client = HarborEnv(base_url=self.options.get("url", "http://127.0.0.1:8200"))
        self.temperature = settings.get("temperature", 1)
        self.indexes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.client.close()

    def sample(self, task, checkpoint, endpoint):
        split = task.openenv_split
        if split not in self.indexes:
            count = self.client.num_tasks(split)
            references = {}
            for start in range(0, count, 50):
                for ref in self.client.get_task_range(split, start, min(start + 50, count)):
                    if ref.task_name in references:
                        raise ValueError(f"OpenEnv split {split} has duplicate task {ref.task_name}")
                    references[ref.task_name] = ref
            self.indexes[split] = references
        ref = self.indexes[split].get(task.task)
        if ref is None:
            raise ValueError(f"OpenEnv split {split} has no task named {task.task}")
        result = self.client.run_rollout(
            split=split, task_index=ref.index, harness=self.options.get("harness", "opencode"),
            sandbox=self.options.get("sandbox", "docker"), reward_key=task.reward_key,
            llm_url=endpoint.url, model=endpoint.model, api_key=endpoint.api_key, purpose="train",
            sampling={"temperature": self.temperature},
            agent_step_limit=self.options.get("agent_step_limit", 0),
        )
        turns = captured_turns(result, self.temperature)
        if result.task_name != ref.task_name or result.task_id != ref.task_id or result.dataset != ref.dataset:
            raise ValueError("OpenEnv rollout task identity does not match the requested task")
        return Rollout(checkpoint, result.reward, turns)


class VLLMPublisher:
    """Serve each immutable student checkpoint under a new model ID."""

    def __init__(self, args, settings):
        del args
        self.options = settings.get("vllm", {})

    @contextmanager
    def serve(self, checkpoint):
        model = f"tianxi-opd-{uuid4().hex[:12]}-{checkpoint.name}"
        host = self.options.get("host", "127.0.0.1")
        health_host = "127.0.0.1" if host == "0.0.0.0" else host
        port = self.options.get("port", 8301)
        endpoint = self.options.get("url", f"http://127.0.0.1:{port}/v1").rstrip("/")
        command = [
            self.options.get("executable", "vllm"), "serve", str(checkpoint),
            "--host", host, "--port", str(port), "--served-model-name", model,
            "--logprobs-mode", "processed_logprobs", "--generation-config", "vllm",
            *self.options.get("args", []),
        ]
        environment = dict(os.environ)
        if "cuda_visible_devices" in self.options:
            environment["CUDA_VISIBLE_DEVICES"] = self.options["cuda_visible_devices"]
        log = checkpoint.parent / f"{checkpoint.name}-vllm.log"
        with log.open("w") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            try:
                deadline = time.monotonic() + self.options.get("startup_timeout_sec", 300)
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        with urlopen(f"http://{health_host}:{port}/v1/models", timeout=2) as response:
                            ready = any(row["id"] == model for row in json.load(response)["data"])
                        if ready:
                            break
                    except URLError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"vLLM failed to publish {checkpoint.name}. See {log}")
                yield Endpoint(endpoint, model)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


class CommandPublisher:
    """Invoke a site-provided checkpoint publisher and read the served model identity."""

    def __init__(self, args, settings):
        del args
        self.options = settings["publisher"]

    @contextmanager
    def serve(self, checkpoint):
        receipt = json.loads(subprocess.check_output(
            [*self.options["command"], str(checkpoint)], text=True,
            timeout=self.options.get("timeout_sec", 600),
        ))
        if Path(receipt["checkpoint"]).resolve() != checkpoint or not receipt["model"]:
            raise ValueError("Checkpoint publisher returned an inconsistent receipt")
        url = receipt["url"]
        location = urlsplit(url)
        if (location.scheme not in ("http", "https") or not location.hostname
                or location.username or location.password or location.query or location.fragment):
            raise ValueError("Checkpoint publisher returned an invalid model endpoint")
        key_env = self.options.get("api_key_env")
        yield Endpoint(url, receipt["model"], os.environ[key_env] if key_env else "")

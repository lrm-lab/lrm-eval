"""Harbor failure diagnostics, dispatch guard, and resume fingerprints."""

from collections import Counter
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile


PAUSE_FILE = ".tianxi-pause.json"
INTEGRITY_FILE = ".tianxi-harbor-integrity.json"
PLUGIN = "batch_guard:DispatchGuard"
PAUSED_EXIT = 75
LOG_LIMIT = 96 * 1024
MODEL_ERRORS_FILE = "model-gateway-errors.json"
MODEL_ERROR_SUMMARIES = {
    "model_gateway_timeout": "模型网关等待上游响应超时。",
    "model_gateway_transport": "模型网关连接上游失败或连接中断。",
    "model_gateway_http_error": "模型网关收到上游 HTTP 错误。",
    "model_gateway_error": "模型网关记录到上游请求失败。",
}

# Only fixed summaries are exported. Raw log and exception text can contain
# prompts, command lines, headers, and credentials.
SIGNATURES = (
    ("model_gateway_timeout", "模型请求等待响应超时。", (
        "engine timed out", "before the model idle timeout", "idle timeout waiting for sse")),
    ("model_gateway_transport", "模型响应流中断。", (
        "stream disconnected before completion", "could not reach engine",
        "server disconnected without sending a response")),
    ("model_gateway_error", "模型请求返回服务端错误。", (
        "reason=server_error", "unexpected status 502 bad gateway",
        "unexpected status 503 service unavailable", "unexpected status 504 gateway timeout")),
    ("agent_tls", "模型请求的 TLS 证书校验失败。", (
        "certificate verify failed", "certificate has expired", "unable to verify the first certificate",
        "self signed certificate", "self-signed certificate", "err_tls_cert", "ip address mismatch")),
    ("agent_protocol", "Agent 使用的模型 API 协议和网关不兼容。", (
        ".responses is not a function", "unsupported api format", "unsupported model api format")),
    ("agent_cli", "Agent CLI 拒绝了启动参数，请核对 harness 版本。", (
        "clierror/showhelp", "unknown option:", "unrecognized flag:", "unrecognized arguments:",
        "usage\n    opencode run [flags]")),
    ("agent_auth", "模型网关拒绝了 Agent 凭据或访问权限。", (
        "status code: 401", "status code: 403", '"statuscode":401', '"statuscode":403',
        "invalid api key", "authentication failed", "missing model credential")),
    ("agent_rate_limit", "模型网关限流，请检查请求配额。", (
        "status code: 429", '"statuscode":429', "rate limit exceeded")),
    ("agent_connect", "Agent 无法从任务沙箱连接模型 API，请检查代理、路由和 DNS。", (
        "cannot connect to api", "unable to connect. is the computer able", "econnrefused",
        "enotfound", "getaddrinfo eai_again", "network is unreachable", "connect timeout")),
    ("agent_install", "Agent 安装或启动依赖失败。", (
        "npm err!", "unable to locate package", "failed to install package", "failed to fetch http",
        "could not resolve deb.debian.org", "opencode: command not found")),
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_integrity(job):
    return {name: file_sha256(job / name) for name in ("config.json", "lock.json")}


def write_private_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def tree_entries(root):
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if path.is_symlink():
                yield relative, mode, os.readlink(path)
            elif path.is_file():
                yield relative, mode, file_sha256(path)
            elif path.is_dir():
                yield relative, mode, None
            else:
                raise ValueError(f"Unrecognized task input file type: {relative}")
        dirs.sort()


def dataset_sha256(dataset, selected):
    names = sorted(set(selected)) if selected else sorted(
        path.parent.name for path in dataset.glob("*/task.toml")
    )
    if not names:
        raise ValueError("No selected Harbor tasks to lock")
    return document_sha256({name: list(tree_entries(dataset / name)) for name in names})


def source_host():
    connection = os.environ.get("SSH_CONNECTION", "").split()
    return connection[2] if len(connection) == 4 else socket.gethostname()


def tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - LOG_LIMIT))
            return stream.read(LOG_LIMIT).decode("utf-8", errors="replace").lower()
    except OSError:
        return ""


def primary_exception(result):
    return result.get("exception_info") or next(
        (step["exception_info"] for step in result.get("step_results") or [] if step.get("exception_info")),
        {},
    )


def diagnostic(kind, summary, source, host=None, **details):
    return {"kind": kind, "summary": summary,
            "raw_log": {"host": host or source_host(), "path": str(source.absolute())},
            **details}


def model_gateway_details(directory, host=None):
    path = directory / MODEL_ERRORS_FILE
    if not path.is_file() or path.is_symlink():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    count = data.get("count")
    responses = data.get("responses")
    if type(count) is not int or count < 1 or type(responses) is not int or responses < 0:
        return None
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in MODEL_ERROR_SUMMARIES:
        kind = "model_gateway_error"
    details = {"count": count, "responses": responses}
    status = data.get("http_status")
    if kind == "model_gateway_http_error" and type(status) is int and 400 <= status <= 599:
        details["http_status"] = status
    session_id = data.get("session_id")
    if isinstance(session_id, str) and 1 <= len(session_id) <= 64 and session_id.isascii() and session_id.isalnum():
        details["session_id"] = session_id
    return diagnostic(kind, MODEL_ERROR_SUMMARIES[kind], path, host, **details)


def missing_model_patch(directory, host=None):
    path = directory / "verifier/test-stdout.txt"
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        return None
    try:
        with path.open("rb") as stream:
            for line in stream.read(LOG_LIMIT).splitlines():
                if line.startswith(b"[verifier] no model.patch submitted"):
                    return diagnostic("missing_model_patch", "评分器未收到补丁，测试使用原始代码。", path, host)
    except OSError:
        pass
    return None


def failure_details(result, directory, host=None, *, model_gateway=None):
    info = primary_exception(result)
    error = info.get("exception_type")
    if not error:
        return model_gateway if model_gateway and model_gateway["responses"] == 0 else None
    paths = [directory / item for item in (
        "agent/opencode.txt", "agent/pi.txt", "agent/openclaw.txt", "agent/codex.txt",
        "agent/claude-code.txt", "agent/agent.log",
        "agent/agent-setup.log", "agent/setup.log", "trial.log",
        "verifier/test-stdout.txt", "verifier/test-stderr.txt", "verifier/test-output.txt",
    )]
    paths = [path for path in paths
             if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve())]
    normalized = error.lower()
    if model_gateway and "agent" in normalized and "setup" not in normalized and "install" not in normalized:
        return model_gateway
    kind, summary = "run_error", "运行异常，请查看执行机原始日志。"
    source = paths[0] if paths else directory / "result.json"
    if "cancel" in normalized:
        kind, summary = "cancelled", "试验已取消，可在核对配置后续跑。"
    elif "timeout" in normalized:
        if "verif" in normalized:
            kind, summary = "verifier_timeout", "评分器运行超时。"
        elif "setup" in normalized or "install" in normalized:
            kind, summary = "agent_setup_timeout", "Agent 安装或启动阶段超时。"
        elif "environment" in normalized or "build" in normalized:
            kind, summary = "environment_timeout", "任务容器构建或启动超时。"
        else:
            kind, summary = "agent_timeout", "Agent 作答阶段超时。"
    elif "verif" in normalized or "rewardfile" in normalized:
        kind, summary = "verifier_error", "评分器执行失败，请查看评分日志。"
        source = next((path for path in paths if path.parent.name == "verifier"), source)
    elif "environment" in normalized or "docker" in normalized or "build" in normalized:
        kind, summary = "environment_error", "容器构建、启动或执行失败。"
        source = directory / "trial.log" if (directory / "trial.log").is_file() else source
    else:
        if "agent" in normalized:
            kind, summary = "agent_exit", "Agent 非零退出，请查看 Agent 原始日志。"
        if "setup" in normalized or "install" in normalized:
            kind, summary = "agent_install", "Agent 安装或启动依赖失败。"
        for path in paths:
            text = tail(path)
            match = next((entry for entry in SIGNATURES if any(token in text for token in entry[2])), None)
            if match:
                kind, summary, _ = match
                source = path
                break
    return diagnostic(kind, summary, source, host)


class DispatchPaused(RuntimeError):
    """Stop queued trials after active trials drain."""


class DispatchGuard:
    """A Harbor job plugin using START and END hooks."""

    def __init__(self, consecutive=3, host=None):
        self.limit = int(consecutive)
        self.host = host
        self.active = set()
        self.last = None
        self.streak = 0
        self.pause = None
        self.initialized = False

    async def on_job_start(self, job):
        self.directory = job.job_dir
        self.wait_for_stop = asyncio.Event()
        job.on_trial_started(self.on_start)
        job.on_trial_ended(self.on_end)

    async def on_job_end(self, job_result):
        pass

    async def on_start(self, event):
        if self.pause:
            # The queue already has a permit, but no sandbox has started. This
            # hook is cancelled by the TaskGroup after the active work drains.
            await self.wait_for_stop.wait()
        if not self.initialized:
            write_private_json(self.directory / INTEGRITY_FILE, native_integrity(self.directory))
            self.initialized = True
        self.active.add(event.trial_name)

    async def on_end(self, event):
        self.active.discard(event.trial_name)
        if self.limit and not self.pause:
            result = event.result.model_dump(mode="json", include={"exception_info", "step_results"})
            failure = failure_details(result, self.directory / event.trial_name, self.host)
            kind = failure["kind"] if failure else None
            if kind == "cancelled":
                return
            key = (kind, primary_exception(result).get("exception_type"))
            self.streak = self.streak + 1 if kind and key == self.last else int(bool(kind))
            self.last = key
            if self.streak >= self.limit and self.has_queued_trials():
                self.pause = {**failure, "consecutive": self.streak, "trial_id": str(event.trial_id)}
                write_private_json(self.directory / PAUSE_FILE, self.pause)
        if self.pause and not self.active:
            raise DispatchPaused(f"Paused after {self.streak} consecutive {self.pause['kind']} failures")

    def has_queued_trials(self):
        try:
            progress = json.loads((self.directory / "result.json").read_text())
        except (OSError, ValueError):
            # Harbor can be updating its progress file from another trial.
            return True
        return (progress.get("stats") or {}).get("n_pending_trials", 0) > 0

"""Run native Harbor trials behind the OpenEnv model capture gateway."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager, contextmanager, nullcontext
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import socket
from uuid import UUID

import batch_guard

TRACE_NAME = "model-gateway.json"


class GatewayTerminated(KeyboardInterrupt):
    """The batch runner received SIGTERM."""


@contextmanager
def upstream_failures():
    from openenv.core.harness.capture.upstream import (
        UpstreamHTTPError, UpstreamTimeoutError, UpstreamTransportError,
    )

    errors = {}

    class Handler(logging.Handler):
        def emit(self, record):
            # OpenEnv logs the original exception before translating the HTTP
            # response for the harness. Keep only its category and status.
            if record.msg != "upstream error [%s]: %s" or not isinstance(record.args, tuple) or len(record.args) != 2:
                return
            session_id, error = record.args
            if isinstance(error, UpstreamHTTPError):
                info = {"kind": "model_gateway_http_error", "http_status": error.status_code}
            elif isinstance(error, UpstreamTimeoutError):
                info = {"kind": "model_gateway_timeout"}
            elif isinstance(error, UpstreamTransportError):
                info = {"kind": "model_gateway_transport"}
            else:
                info = {"kind": "model_gateway_error"}
            errors[session_id] = info

    logger = logging.getLogger("openenv.core.harness.capture.server")
    handler = Handler()
    logger.addHandler(handler)
    try:
        yield errors
    finally:
        logger.removeHandler(handler)


def capture_document(path, trial_id):
    data = path.read_bytes()
    doc = json.loads(data)
    if (doc.get("schema_version") != 1 or doc.get("trial_id") != trial_id
            or doc.get("purpose") != "eval" or type(doc.get("validated")) is not bool
            or not isinstance(doc.get("turns"), list)):
        raise ValueError(f"Invalid gateway capture for trial {trial_id}")
    turns = doc["turns"]
    if any(not isinstance(turn, dict) or not isinstance(turn.get("request_messages"), list)
           for turn in turns):
        raise ValueError(f"Invalid model request window for trial {trial_id}")
    return doc, data


def has_tool_round_trip(doc):
    calls = set()
    for turn in doc["turns"]:
        for message in turn["request_messages"]:
            if (isinstance(message, dict) and message.get("role") == "tool"
                    and message.get("tool_call_id") in calls):
                return True
        for call in (turn.get("response_message") or {}).get("tool_calls") or ():
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                calls.add(call["id"])
    return False


@asynccontextmanager
async def serve_gateway(host, base_url, model, key):
    import uvicorn
    from openenv.core.harness.capture.server import create_app

    app = create_app(llm_url=base_url, model=model, api_key=key, capture_level="text",
                     max_output_tokens=None, admin_key=secrets.token_urlsafe(32))
    with socket.socket() as listener, upstream_failures() as errors:
        listener.bind((host, 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
        # The runner owns shutdown so it can publish completed trials first.
        if hasattr(server, "capture_signals"):
            server.capture_signals = nullcontext
        else:
            server.install_signal_handlers = lambda: None
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(15):
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("Capture gateway exited during startup")
                    await asyncio.sleep(0.05)
            yield app, f"http://{host}:{port}", errors
        finally:
            server.should_exit = True
            await serving


async def capture_batch(args, task_paths, job, base_url, model, key, write_capture, on_trial_finished):
    from openenv.core.harness.capture.export import export_session
    from openenv.harbor.rollout import run_rollout

    counts = Counter()
    tasks = iter(task for task in task_paths for _ in range(args.attempts))
    planned = len(task_paths) * args.attempts
    last_failure = None
    streak = 0
    async with serve_gateway(args.capture_host, base_url, model, key) as (app, url, errors):
        async def run_one(task):
            trial_path = None
            trial_id = None
            session = None

            def remember_trial(trial):
                nonlocal trial_path, trial_id
                trial_id = str(trial.id)
                UUID(trial_id)
                trial_path = trial.paths.trial_dir

            def remember_session(session_id):
                nonlocal session
                session = app.state.registry.get(session_id)

            result = await run_rollout(
                task_dir=task, harness=args.agent, sandbox="docker",
                registry=app.state.registry, inference=app.state.inference,
                intercept_url=url, local_intercept_url=url, model=model,
                trials_dir=job, reward_key=args.reward_key, require_reward=False,
                capture_level="text", purpose="eval", on_trial_created=remember_trial,
                on_session_created=remember_session,
            )
            turns = []
            valid = False
            if session is not None:
                export = export_session(session, include_messages=True, include_discarded=True, capture_level="text")
                fields = ("node_id", "index", "root_id", "discarded", "request_messages", "request_tools",
                          "response_message", "finish_reason", "sampling_params", "requested_sampling_params")
                turns = [{field: turn[field] for field in fields} for turn in export["turns"]]
                valid = bool(turns) and not any("[FATAL" in finding for finding in export["validation"])
            failure = result.exception_type
            upstream = errors.pop(session.session_id, {}) if session is not None else {}
            receipt = trial_path / "result.json" if trial_path is not None else None
            if receipt is not None and receipt.is_file():
                if session is not None and session.upstream_errors:
                    info = {**upstream, "session_id": session.session_id,
                            "count": session.upstream_errors, "responses": len(turns)}
                    write_capture(trial_path / batch_guard.MODEL_ERRORS_FILE, info)
                    kind = upstream.get("kind", "model_gateway_error")
                    status = f" (HTTP {upstream['http_status']})" if "http_status" in upstream else ""
                    print(f"{task.name}: {session.upstream_errors} upstream request error(s), latest {kind}{status}",
                          flush=True)
                    if not turns:
                        failure = kind
                document = {
                    "schema_version": 1, "trial_id": trial_id, "purpose": "eval",
                    "capture_level": result.capture_level, "upstream_model": model,
                    "validated": valid, "atif_reconciliation": result.atif,
                    "model_calls": session.model_calls if session else 0,
                    "turns": turns,
                }
                await asyncio.to_thread(write_capture, trial_path / TRACE_NAME, document)
                failure = await asyncio.to_thread(on_trial_finished, receipt) or failure
            elif upstream:
                failure = upstream["kind"]
            print(f"{task.name}: captured {len(turns)} model responses"
                  f", capture validated={valid}, reward={None if failure else result.reward}", flush=True)
            if not failure and valid and result.reward is not None:
                counts["captured"] += 1
                return None
            return failure or ("unscored_trial" if valid else "invalid_gateway_capture")

        async def worker():
            nonlocal last_failure, streak
            while not counts["paused"]:
                task = next(tasks, None)
                if task is None:
                    return
                counts["started"] += 1
                failure = await run_one(task)
                streak = streak + 1 if failure and failure == last_failure else int(bool(failure))
                last_failure = failure
                if failure:
                    counts["last_failure"] = failure
                    print(f"Gateway diagnostic: {failure}", flush=True)
                if args.pause_after and streak >= args.pause_after and counts["started"] < planned:
                    counts["paused"] = 1

        async with asyncio.TaskGroup() as group:
            for _ in range(min(args.concurrent, planned)):
                group.create_task(worker())
    return counts


def run_capture(args, task_paths, job, base_url, model, key, write_capture, on_trial_finished):
    try:
        address = ipaddress.IPv4Address(args.capture_host)
    except ipaddress.AddressValueError as exc:
        raise ValueError("Set --capture-host to the Docker-reachable IPv4 address of this host") from exc
    if address.is_unspecified or address.is_loopback:
        raise ValueError("--capture-host must be a Docker-reachable address, such as 172.17.0.1")
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(filter(None, (os.environ.get(name), args.capture_host)))
    if args.agent_version:
        os.environ["OPENENV_HARBOR_AGENT_VERSIONS"] = json.dumps({args.agent: args.agent_version})
    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        terminating = False

        def terminate():
            nonlocal terminating
            terminating = True
            task.cancel()

        loop.add_signal_handler(signal.SIGTERM, terminate)
        try:
            return await capture_batch(args, task_paths, job, base_url, model, key, write_capture, on_trial_finished)
        except asyncio.CancelledError:
            if terminating:
                raise GatewayTerminated from None
            raise
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    try:
        return asyncio.run(run())
    except ModuleNotFoundError as exc:
        raise ValueError(f"Gateway mode requires OpenEnv with Harbor support ({exc.name})") from exc

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework"))

import batch_guard
import gateway_capture
import report
import run


class GatewayPublicationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "export"
        self.job = self.root / "jobs" / "batch"
        self.metadata = {"dataset": "deepswe-1.1", "model": "vllm/test-model",
                         "batch": "batch", "harness": "codex"}

    def write(self, path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))

    def receipt(self, task, reward, **extra):
        path = self.job / task / "result.json"
        self.write(path, {"id": str(uuid4()), "task_name": task,
                          "finished_at": "2026-11-25T00:00:00Z",
                          "verifier_result": {"rewards": {"reward": reward}}, **extra})
        return path

    def test_export_keeps_valid_scores_and_exposes_patch_and_gateway_diagnostics(self):
        no_patch = self.receipt("no-patch", 0)
        log = no_patch.parent / "verifier" / "test-stdout.txt"
        log.parent.mkdir()
        log.write_text("[verifier] no model.patch submitted — grading pristine base state\n")
        recovered = self.receipt("recovered", 1)
        failed = self.receipt("failed", 0)
        for receipt, responses in ((recovered, 3), (failed, 0)):
            self.write(receipt.parent / batch_guard.MODEL_ERRORS_FILE,
                       {"count": 2, "responses": responses, "kind": "model_gateway_http_error",
                        "http_status": 503, "message": "private diagnostic text"})
        self.write(self.output / "batch-status.json",
                   {"status": "running", "completed": 0, "planned": 4,
                    "source": {"host": "runner", "path": str(self.job)}})
        run.publish(self.job, self.output, self.metadata, source_host="runner",
                    status_updates={}, mark_interrupted=True)
        _, records, _, status = report.read_batch(self.output)
        rows = {row["task_id"]: row for row in records}
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["completed"], 3)
        self.assertEqual(rows["no-patch"]["status"], "scored")
        self.assertEqual(rows["no-patch"]["reward"], 0)
        self.assertEqual(rows["no-patch"]["warnings"][0]["kind"], "missing_model_patch")
        self.assertEqual(rows["recovered"]["reward"], 1)
        self.assertEqual(rows["recovered"]["warnings"][0]["http_status"], 503)
        self.assertEqual(rows["failed"]["status"], "failed")
        self.assertIsNone(rows["failed"]["reward"])
        self.assertEqual(rows["failed"]["verifier_rewards"]["reward"], 0)
        self.assertEqual(rows["failed"]["failure"]["kind"], "model_gateway_http_error")
        self.assertEqual(rows["failed"]["failure"]["http_status"], 503)
        self.assertNotIn("private diagnostic text", (self.output / "episodes.jsonl").read_text())
        summary = json.loads((self.output / "batch-report.json").read_text())
        self.assertEqual(summary["totals"]["scored"], 2)
        self.assertEqual(summary["warning_kinds"]["missing_model_patch"], 1)
        self.assertIn("运行故障 model_gateway_http_error 1", report.as_markdown(summary))

    def test_legacy_agent_logs_classify_model_timeout(self):
        receipt = self.receipt("timed-out", 0, exception_info={"exception_type": "NonZeroAgentExitCodeError"})
        log = receipt.parent / "agent" / "codex.txt"
        log.parent.mkdir()
        log.write_text('{"type":"error","message":"idle timeout waiting for SSE"}\n')
        run.publish(self.job, self.output, self.metadata, source_host="runner")
        _, rows, _, _ = report.read_batch(self.output)
        self.assertEqual(rows[0]["status"], "timeout")
        self.assertEqual(rows[0]["failure"]["kind"], "model_gateway_timeout")
        self.assertIsNone(rows[0]["reward"])

    def test_gateway_publishes_before_exit_and_keeps_results_on_interrupt(self):
        dataset = self.root / "dataset"
        for task in ("first", "second"):
            path = dataset / task / "task.toml"
            path.parent.mkdir(parents=True)
            path.write_text("")
        args = SimpleNamespace(model="vllm/test-model", task=[], plan=False, base_url="https://localhost",
                               work=self.root, batch="batch", source_host="runner", attempts=1,
                               reward_key="reward")

        def interrupted(args, tasks, job, url, model, key, write, on_trial_finished):
            on_trial_finished(self.receipt("first", 1))
            _, records, _, state = report.read_batch(self.output)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["completed"], 1)
            self.assertEqual(records[0]["reward"], 1)
            self.receipt("second", 0)
            raise gateway_capture.GatewayTerminated

        with patch.object(gateway_capture, "run_capture", side_effect=interrupted):
            self.assertEqual(run.execute_gateway(args, dataset, self.output, self.metadata), 143)
        _, records, _, state = report.read_batch(self.output)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["completed"], 2)
        self.assertEqual([row["reward"] for row in records], [1, 0])

    def test_gateway_translates_sigterm_to_interruption(self):
        import asyncio

        async def interrupt(*args):
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.1)
            self.fail("Gateway task was not cancelled")

        args = SimpleNamespace(capture_host="172.17.0.1", agent_version=None)
        with patch.dict(os.environ), patch.object(gateway_capture, "capture_batch", side_effect=interrupt):
            with self.assertRaises(gateway_capture.GatewayTerminated):
                gateway_capture.run_capture(args, [], self.job, "", "", None, None, None)


class GatewayCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_upstream_http_status_is_written_before_publication(self):
        class HTTPError(Exception):
            status_code = 503

        upstream = SimpleNamespace(UpstreamHTTPError=HTTPError, UpstreamTimeoutError=TimeoutError,
                                   UpstreamTransportError=ConnectionError)
        session = SimpleNamespace(session_id="stest", upstream_errors=1, model_calls=1)
        app = SimpleNamespace(state=SimpleNamespace(registry={"stest": session}, inference=None))
        logger = logging.getLogger("openenv.core.harness.capture.server")
        trial_id = uuid4()

        @asynccontextmanager
        async def gateway(*args):
            with gateway_capture.upstream_failures() as errors:
                yield app, "http://localhost", errors

        async def rollout(**kwargs):
            kwargs["on_trial_created"](SimpleNamespace(id=trial_id, paths=SimpleNamespace(trial_dir=job)))
            kwargs["on_session_created"](session.session_id)
            logger.warning("upstream error [%s]: %s", session.session_id, HTTPError("private upstream text"))
            return SimpleNamespace(exception_type=None, reward=0, capture_level="text", atif=None)

        def publish(receipt):
            diagnostic = batch_guard.model_gateway_details(receipt.parent)
            self.assertEqual(diagnostic["http_status"], 503)
            self.assertEqual(diagnostic["responses"], 0)
            self.assertEqual(diagnostic["session_id"], "stest")
            self.assertNotIn("private upstream text", json.dumps(diagnostic))

        modules = {
            "openenv.core.harness.capture.upstream": upstream,
            "openenv.core.harness.capture.export": SimpleNamespace(
                export_session=lambda *args, **kwargs: {"turns": [], "validation": []}),
            "openenv.harbor.rollout": SimpleNamespace(run_rollout=rollout),
        }
        args = SimpleNamespace(attempts=1, capture_host="172.17.0.1", agent="codex",
                               reward_key="reward", concurrent=1, pause_after=1)
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "result.json").write_text("{}")
            with patch.dict(sys.modules, modules), patch.object(gateway_capture, "serve_gateway", gateway):
                with self.assertLogs(logger, logging.WARNING):
                    counts = await gateway_capture.capture_batch(
                        args, [Path("first"), Path("second")], job, "", "", None,
                        lambda path, value: path.write_text(json.dumps(value)), publish)
        self.assertEqual(counts["started"], 1)
        self.assertEqual(counts["paused"], 1)
        self.assertEqual(counts["last_failure"], "model_gateway_http_error")


if __name__ == "__main__":
    unittest.main()

"""Build and exercise stateful ToolSpace tasks using Harbor's writer/checker hooks."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from synthesis_methods import Case
from synthesis_reasoning import toolspace as catalog_case
from synthesis_reasoning_audit import tool_answer, toolspace as legacy_audit

HERE = Path(__file__).resolve().parent


@dataclass
class ToolCase(Case):
    service: dict | None = None


def toolspace(split, index, seed, *, graph_file=None):
    case = catalog_case(split, index, seed, graph_file=graph_file)
    request = json.loads(case.files["request.json"])
    private = json.loads(case.files.pop("tool-data.json"))
    reserved = {"status", "prepare", "assign", "refer", "authorize"}
    if reserved.intersection(private["catalog"]):
        raise ValueError("Tool identifiers conflict with workflow action names")
    mode = (index - 1) % 3
    receipt = f"T-{random.Random(str(seed)).getrandbits(96):024x}"
    action = "assign" if mode == 0 else "refer"
    private.update(request, action=action, stale_once=mode == 2, receipt=receipt)
    case.expected["rows"].update(action=action, receipt=receipt)
    case.files["tools.py"] = (HERE / "synthesis_tool_client.py").read_text()
    case.rules += """
Complete the handoff using the same CLI. Run 'prepare DESTINATION_ID' to obtain the required
action and version. For a direct handoff run 'assign DESTINATION_ID VERSION'. For a referral,
run 'authorize DESTINATION_ID VERSION', then 'refer DESTINATION_ID VERSION AUTHORIZATION'.
On a stale version prepare again and obtain authorization for the new version. Repeating a committed action with its current version returns the same receipt.
Include action and receipt in answer.json as well as record_ids and label. The live service
must show the corresponding completed handoff. The 'status' command shows the committed state."""
    case.reference += """
target = answer['record_ids'][-1]
def commit():
    prepared = call('prepare', target)
    version = str(prepared['version'])
    if prepared['action'] == 'refer':
        token = call('authorize', target, version)['authorization']
        return call('refer', target, version, token)
    return call('assign', target, version)
result = commit()
if result.get('error') == 'stale_version':
    result = commit()
answer.update(action=result['action'], receipt=result['receipt'])
(app / 'answer.json').write_text(json.dumps(answer))
"""
    case.environment = ("tool-assign", "tool-refer", "tool-recover")[mode]
    case.wrong = [dict(case.expected["rows"], action="unknown"),
                  dict(case.expected["rows"], record_ids=case.expected["rows"]["record_ids"][1:])]
    return ToolCase(**vars(case), service=private)


def write(folder, method, split, case, scenario):
    from synthesize import write_case
    write_case(folder, method, split, case, scenario)
    env = folder / "environment"
    service = env / "catalog"
    service.mkdir()
    (service / "fixture.json").write_text(json.dumps(case.service))
    shutil.copyfile(HERE / "synthesis_tool_service.py", service / "serve.py")
    (service / "Dockerfile").write_text(
        'FROM python:3.11-slim\nWORKDIR /srv\nCOPY serve.py fixture.json ./\n'
        'CMD ["python3", "serve.py", "fixture.json", "8765"]\n')
    (env / ".dockerignore").write_text("catalog/\n")
    (env / "docker-compose.yaml").write_text(
        'services:\n  catalog:\n    build:\n      context: ${CONTEXT_DIR}/catalog\n'
        '    network_mode: "service:main"\n'
        '    healthcheck:\n'
        """      test: ["CMD", "python3", "-c", "import socket; socket.create_connection(('127.0.0.1',8765),1)"]\n"""
        '      interval: 1s\n      timeout: 2s\n      retries: 15\n')
    tests = folder / "tests"
    (tests / "verify.py").rename(tests / "output_score.py")
    shutil.copyfile(HERE / "synthesis_tool_grade.py", tests / "verify.py")


def audit(inputs):
    fixture = inputs.parent / "catalog" / "fixture.json"
    if not fixture.exists():
        return legacy_audit(inputs)
    data = json.loads(fixture.read_text())
    request = json.loads((inputs / "request.json").read_text())
    return dict(tool_answer(request, data), action=data["action"], receipt=data["receipt"])


def check(task, scratch, auditor):
    from synthesize import validate_case
    env = task / "environment"
    fixture = env / "catalog" / "fixture.json"
    if not fixture.exists():
        return validate_case(task, scratch, auditor)
    app = scratch / "app"
    shutil.copytree(env / "inputs", app / "inputs")
    tests = task / "tests"
    target = tests / "expected.json"
    expected = json.loads(target.read_text())
    answer = app / expected["file"]
    reward = scratch / "reward.txt"
    with subprocess.Popen([sys.executable, "-B", str(env / "catalog" / "serve.py"), str(fixture), "0"],
                          stdout=subprocess.PIPE, text=True) as service:
        try:
            port = str(int(service.stdout.readline()))
            def grade(wanted):
                reward.unlink(missing_ok=True)
                subprocess.run([sys.executable, "-B", str(tests / "verify.py"), str(app), str(target),
                                str(reward), port], check=True, capture_output=True, timeout=10)
                if int(reward.read_text()) != wanted:
                    raise ValueError(f"Unexpected ToolSpace score (expected {wanted})")
            grade(0)
            answer.write_text(json.dumps(expected["rows"]))
            grade(0)  # The correct JSON alone cannot complete the handoff.
            answer.unlink()
            subprocess.run([sys.executable, "-B", str(task / "solution" / "solve.py"), str(app)],
                           env={**os.environ, "TOOLSPACE_PORT": port}, check=True, timeout=20, capture_output=True)
            grade(1)
            if auditor and auditor(env / "inputs") != expected["rows"]:
                raise ValueError("ToolSpace reference differs from independent graph traversal")
            for wrong in json.loads((tests / "typical_wrong.json").read_text()):
                answer.write_text(json.dumps(wrong))
                grade(0)
        finally:
            service.terminate()

"""Grade the submitted answer and the committed state of the trial service."""

import json
from pathlib import Path
import sys
from urllib.request import build_opener, ProxyHandler, Request

from output_score import score


if __name__ == "__main__":
    app, target, reward = map(Path, sys.argv[1:4])
    expected = json.loads(target.read_text())
    port = int(sys.argv[4]) if len(sys.argv) == 5 else 8765
    request = Request(f"http://127.0.0.1:{port}/call", data=b'["status"]')
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        actual = json.load(response)["completed"]
    goal = expected["rows"]
    complete = {"destination": goal["record_ids"][-1], "action": goal["action"], "receipt": goal["receipt"]}
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text(str(int(actual == complete and score(app, expected))) + "\n")

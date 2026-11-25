"""Private per-trial ToolSpace service. Only its JSON interface is shared with the agent."""

import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys


class Catalog:
    def __init__(self, fixture):
        self.data = fixture
        self.available = {(fixture["start_tool"], fixture["start_id"])}
        self.visited = set()
        self.prepared = None
        self.current = 1
        self.committed = None
        self.authorization = None

    def call(self, args):
        data = self.data
        if not args:
            return data["catalog"]
        if args == ["status"]:
            return {"completed": self.committed}
        if len(args) == 2 and tuple(args) in self.available:
            tool, identifier = args
            row = data["records"][tool][identifier]
            for link, target in data["catalog"][tool]["links"].items():
                self.available.add((target, row["links"][link]))
            self.visited.add((tool, identifier))
            return row
        if len(args) == 2 and args[0] == "prepare":
            target = args[1]
            if (data["goal_tool"], target) not in self.visited:
                raise ValueError("look up the destination first")
            self.prepared = target
            return {"version": self.current, "action": data["action"]}
        if len(args) == 3 and args[0] == "authorize":
            _, target, version = args
            if data["action"] != "refer" or self.prepared != target or version != str(self.current):
                raise ValueError("prepare the current referral before requesting authorization")
            key = f"{data['receipt']}:{target}:{version}".encode()
            self.authorization = hashlib.sha256(key).hexdigest()[:24]
            return {"authorization": self.authorization}
        if args and args[0] in ("assign", "refer") and len(args) in (3, 4):
            action, target, version = args[:3]
            if action != data["action"] or self.prepared != target:
                raise ValueError("prepare the destination and use its required action")
            if action == "refer" and (len(args) != 4 or args[3] != self.authorization):
                raise ValueError("referral requires the current authorization")
            if action == "assign" and len(args) != 3:
                raise ValueError("assign takes a destination and version")
            if version != str(self.current):
                raise ValueError("stale version")
            if data["stale_once"] and self.current == 1:
                self.current += 1
                self.prepared = None
                self.authorization = None
                return {"error": "stale_version", "recovery": "prepare again"}
            self.committed = {"destination": target, "action": action, "receipt": data["receipt"]}
            return self.committed
        raise ValueError("unknown request or undiscovered identifier")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.path != "/call" or not 0 < size <= 8192:
                raise ValueError("invalid tool request")
            args = json.loads(self.rfile.read(size))
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise ValueError("expected a list of strings")
            reply = self.server.catalog.call(args)
            status = 200
        except (ValueError, RecursionError) as error:
            reply = {"error": str(error)}
            status = 400
        body = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    fixture, port = sys.argv[1:]
    with HTTPServer(("127.0.0.1", int(port)), Handler) as server:
        server.catalog = Catalog(json.loads(Path(fixture).read_text()))
        print(server.server_port, flush=True)
        server.serve_forever()

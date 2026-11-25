"""List the catalog or invoke a tool with its arguments."""

import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler, Request


if __name__ == "__main__":
    port = int(os.environ.get("TOOLSPACE_PORT", "8765"))
    request = Request(f"http://127.0.0.1:{port}/call", data=json.dumps(sys.argv[1:]).encode())
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            print(response.read().decode())
    except HTTPError as error:
        sys.exit(error.read().decode())

"""Read-only CLI for the tool catalogs emitted by graph-based synthesis."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", nargs="?")
    parser.add_argument("identifier", nargs="?")
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name("tool-data.json").read_text())
    if args.tool is None:
        print(json.dumps(data["catalog"]))
        return
    try:
        row = data["records"][args.tool][args.identifier]
    except KeyError:
        parser.error("unknown tool or record identifier")
    print(json.dumps(row))


if __name__ == "__main__":
    main()

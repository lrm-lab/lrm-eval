"""Standalone verifier copied into each generated Harbor task."""

import csv
import json
from pathlib import Path
import sys


def score(app, expected):
    candidate = app / expected["file"]
    try:
        if candidate.stat().st_size > 2_000_000 or not candidate.is_file():
            return 0
        with candidate.open(encoding="utf-8", newline="") as handle:
            if expected["kind"] == "csv":
                header, *rows = csv.reader(handle, strict=True)
                return int(header == expected["columns"] and rows == expected["rows"])
            if expected["kind"] == "json":
                rows = json.load(handle)
                # Serializing the parsed value distinguishes booleans and floats from integers.
                return int(json.dumps(rows, sort_keys=True) == json.dumps(expected["rows"], sort_keys=True))
            raise RuntimeError(f"Unknown verifier format {expected['kind']}")
    except (FileNotFoundError, ValueError, UnicodeError, csv.Error, IsADirectoryError):
        return 0


def main():
    app, target, reward = map(Path, sys.argv[1:])
    expected = json.loads(target.read_text())
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text(str(score(app, expected)) + "\n")


if __name__ == "__main__":
    main()

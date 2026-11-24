"""Independent fixture checks for the built-in synthesis methods."""

import csv
from datetime import datetime, timezone
from itertools import groupby
import json
import sqlite3


def order_rows(inputs):
    settings = json.loads((inputs / "rules.json").read_text())
    if (inputs / "orders.jsonl").exists():
        orders = [tuple(row[key] for key in ("order_id", "currency", "billed_minor"))
                  for row in map(json.loads, (inputs / "orders.jsonl").read_text().splitlines())]
    else:
        with (inputs / "orders.csv").open(newline="") as file:
            orders = list(csv.reader(file))[1:]
    with (inputs / "ledger.csv").open(newline="") as file:
        ledger = list(csv.reader(file))[1:]
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
            CREATE TABLE orders (id TEXT, currency TEXT, billed INTEGER);
            CREATE TABLE ledger (id TEXT, order_id TEXT, currency TEXT, kind TEXT,
                                 amount INTEGER, state TEXT, stamp TEXT);
            CREATE TABLE signs (kind TEXT, sign INTEGER);
        """)
        db.executemany("INSERT INTO orders VALUES (?, ?, ?)", orders)
        db.executemany("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?)", ledger)
        db.executemany("INSERT INTO signs VALUES (?, ?)", settings["polarity"].items())
        direction = "DESC" if settings["keep"] == "last" else "ASC"
        rows = db.execute(f"""
            WITH ranked AS (
                SELECT ledger.order_id, amount * sign AS value,
                       ROW_NUMBER() OVER (PARTITION BY ledger.id ORDER BY ledger.rowid {direction}) AS rank
                  FROM ledger JOIN orders ON ledger.order_id = orders.id AND ledger.currency = orders.currency
                  JOIN signs ON ledger.kind = signs.kind
                 WHERE state = 'posted' AND julianday(stamp) <= julianday(?)
            ), totals AS (
                SELECT orders.id, currency, billed, COALESCE(SUM(value), 0) AS net
                  FROM orders LEFT JOIN ranked ON orders.id = ranked.order_id AND ranked.rank = 1
                 GROUP BY orders.id, currency, billed
            )
            SELECT id, currency, billed, net, net - billed,
                   CASE WHEN net = billed THEN 'matched' WHEN net < billed THEN 'short' ELSE 'over' END
              FROM totals ORDER BY id
        """, (settings["cutoff"],)).fetchall()
    return [[str(value) for value in row] for row in rows]


def log_rows(inputs):
    settings = json.loads((inputs / "rules.json").read_text())
    field = settings["error_field"]
    rows = []
    for path in sorted(inputs.glob("part-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
                stamp = datetime.fromisoformat(event["ts"])
                service, duration, name, signal = event["service"], event["duration_ms"], event["id"], event[field]
            except (KeyError, TypeError, ValueError):
                continue
            if not (isinstance(service, str) and service and isinstance(name, str) and name
                    and type(duration) is int and duration >= 0 and stamp.tzinfo is not None):
                continue
            if field == "level":
                if signal not in ("INFO", "ERROR"):
                    continue
                failed = signal == "ERROR"
            else:
                if type(signal) is not int or not 100 <= signal <= 599:
                    continue
                failed = 500 <= signal
            utc = stamp.astimezone(timezone.utc)
            label = utc.date().isoformat()
            if settings["bucket"] == "hour":
                label += f"T{utc.hour:02}:00Z"
            rows.append((name, label, service, duration, failed))
    if settings["keep"] == "first":
        rows.reverse()
    final_by_id = {row[0]: row[1:] for row in rows}
    results = []
    for (bucket, service), events in groupby(sorted(final_by_id.values()), lambda row: row[:2]):
        events = list(events)
        n = len(events)
        index = (n * 95 - 1) // 100
        results.append({"bucket": bucket, "service": service, "count": n,
                        "errors": sum(row[3] for row in events), "p95_ms": events[index][2]})
    return results

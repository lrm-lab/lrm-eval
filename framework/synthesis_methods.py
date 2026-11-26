"""Deterministic fixture generators for the first Tianxi synthesis methods."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import io
import json
import random


@dataclass
class Case:
    files: dict[str, str]
    rules: str
    expected: dict
    wrong: list
    reference: str
    scenario: str
    environment: str = "offline-files"
    provenance: dict | None = None


def csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def json_lines(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def shuffled(rng, rows):
    return rng.sample(rows, len(rows))


def order_reconcile(split, index, seed):
    holdout = split == "holdout"
    rng = random.Random(seed)
    cutoff = datetime(2025, (index % 12) + 1, rng.randrange(10, 23), tzinfo=timezone.utc)
    before = (cutoff - timedelta(seconds=1)).isoformat()
    after = (cutoff + timedelta(seconds=1)).isoformat()
    polarity = {"charge": 1, "refund": -1}
    if holdout:
        polarity["chargeback"] = -1
    orders, events, expected = [], [], []
    for number in range(1, rng.randrange(35, 61)):
        order_id = f"O-{index:02}-{number:03}"
        currency = rng.choice(("EUR", "USD", "JPY"))
        billed = rng.randrange(1000, 90000)
        refund = rng.randrange(17, 900)
        balance = 0 if number == 1 else billed + rng.choice((0, 0, -61, 57, -94, 21))
        orders.append((order_id, currency, billed))
        status = "matched" if billed == balance else "short" if balance < billed else "over"
        expected.append([order_id, currency, str(billed), str(balance), str(balance - billed), status])
        if balance:
            events.extend([
                [f"C-{number}", order_id, currency, "charge", balance + refund, "posted", before],
                [f"R-{number}", order_id, currency, "chargeback" if holdout and number % 3 == 0 else "refund",
                 refund, "posted", cutoff.isoformat()],
            ])
        events.extend([
            [f"VOID-{number}", order_id, currency, "charge", 108, "void", before],
            [f"LATE-{number}", order_id, currency, "refund", 107, "posted", after],
            [f"CURR-{number}", order_id, "GBP", "charge", 101, "posted", before],
        ])
    events = shuffled(rng, events)
    duplicates = [
        [row[0], row[1], row[2], row[3], row[4] + 191, row[5], row[6]]
        for row in events if row[0].startswith("C-") and rng.random() < 0.55
    ]
    events = duplicates + events if holdout else events + duplicates
    events.append(["ORPHAN", "missing", "USD", "charge", 999, "posted", before])
    order_columns = ("order_id", "currency", "billed_minor")
    ledger_columns = ("event_id", "order_id", "currency", "kind", "amount_minor", "state", "event_time")
    output_columns = ["order_id", "currency", "billed_minor", "net_minor", "delta_minor", "status"]
    inputs = {
        "ledger.csv": csv_text(ledger_columns, events),
        "rules.json": json.dumps({"cutoff": cutoff.isoformat(), "keep": "last" if holdout else "first",
                                  "polarity": polarity}, indent=2) + "\n",
    }
    if holdout:
        inputs["orders.jsonl"] = json_lines(dict(zip(order_columns, row)) for row in shuffled(rng, orders))
    else:
        inputs["orders.csv"] = csv_text(order_columns, shuffled(rng, orders))
    order_file = "orders.jsonl (JSON object per line)" if holdout else "orders.csv"
    rules = f"""Inputs are in `/app/inputs`. Read `{order_file}`, `ledger.csv`, and `rules.json`.
Every order has one currency. Keep ledger rows whose order exists, currency matches, kind is in
`polarity`, state is `posted`, and event_time is at or before the timezone-aware cutoff.
After filtering, deduplicate across the entire ledger by event_id. Keep the {inputs_preference(holdout)}
eligible occurrence in CSV order, using the amount and order from that row.
Sum amount_minor times the configured polarity for every order, including orders with no eligible event.
Compute delta_minor as net_minor minus billed_minor. Use status `matched` if zero, `short` if
negative, and `over` if positive. Amounts are integer minor units and currencies are never converted.
Write `/app/reconciliation.csv` with header `{','.join(output_columns)}` in order_id ascending order.
Use the exact status values and do not add columns."""
    wrong = [[*expected[0][:3], str(int(expected[0][3]) + 1), *expected[0][4:]], *expected[1:]]
    return Case(inputs, rules, {"file": "reconciliation.csv", "kind": "csv",
                                "columns": output_columns, "rows": expected},
                [wrong, expected[:-1]], ORDER_REFERENCE,
                "Reconcile a processor's settlement export against the order ledger for a reporting close.")


def inputs_preference(holdout):
    return "last" if holdout else "first"


ORDER_REFERENCE = """\
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

app = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
inputs = app / 'inputs'
settings = json.loads((inputs / 'rules.json').read_text())
file = inputs / 'orders.jsonl'
if file.is_file():
    rows = [json.loads(line) for line in file.read_text().splitlines()]
else:
    with (inputs / 'orders.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
orders = {row['order_id']: row for row in rows}
limit = datetime.fromisoformat(settings['cutoff'])
seen = {}
with (inputs / 'ledger.csv').open(newline='') as handle:
    for row in csv.DictReader(handle):
        order = orders.get(row['order_id'])
        sign = settings['polarity'].get(row['kind'])
        if (order is None or order['currency'] != row['currency'] or sign is None
                or row['state'] != 'posted' or datetime.fromisoformat(row['event_time']) > limit):
            continue
        if row['event_id'] not in seen or settings['keep'] == 'last':
            seen[row['event_id']] = row['order_id'], sign * int(row['amount_minor'])
net = dict.fromkeys(orders, 0)
for order_id, amount in seen.values():
    net[order_id] += amount
with (app / 'reconciliation.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(('order_id', 'currency', 'billed_minor', 'net_minor', 'delta_minor', 'status'))
    for order_id, order in sorted(orders.items()):
        billed = int(order['billed_minor'])
        delta = net[order_id] - billed
        status = 'over' if delta > 0 else 'short' if delta < 0 else 'matched'
        writer.writerow((order_id, order['currency'], billed, net[order_id], delta, status))
"""


def log_rollup(split, index, seed):
    holdout = split == "holdout"
    rng = random.Random(seed)
    start = datetime(2025, index % 12 + 1, rng.randrange(5, 23), 23, 55, tzinfo=timezone.utc)
    groups = {}
    records = []
    for number in range(rng.randrange(95, 181)):
        observed = start + timedelta(minutes=rng.randrange(-10, 165 if holdout else 2900))
        offset = timezone(timedelta(hours=rng.choice((-8, 0, 5, 8))))
        service = rng.choice(("checkout", "shipper", "catalog", "revops"))
        duration = rng.randrange(1, 4900)
        failed = rng.randrange(5) == 0
        event = {"id": f"E-{index:02}-{number:04}", "ts": observed.astimezone(offset).isoformat(),
                 "service": service, "duration_ms": duration}
        event.update({"http_status": 503 if failed else 201} if holdout else {"level": "ERROR" if failed else "INFO"})
        records.append(event)
        bucket = observed.strftime("%Y-%m-%dT%H:00Z" if holdout else "%Y-%m-%d")
        groups.setdefault((bucket, service), []).append((duration, failed))
    decoys = [dict(event, duration_ms=event["duration_ms"] + 607) for event in records[::7]]
    records = shuffled(rng, records)
    stream = ((decoys + records) if holdout else (records + decoys))
    body = [json.dumps(event) for event in stream]
    body.insert(len(body) // 3, '{"id":"malformed",')
    body.insert(len(body) // 2, '{"id":"invalid","duration_ms":-1}')
    shards = {}
    for n in range(3):
        start_row, end_row = len(body) * n // 3, len(body) * (n + 1) // 3
        shards[f"part-{n:02}.jsonl"] = "\n".join(body[start_row:end_row]) + "\n"
    shards["rules.json"] = json.dumps({"keep": inputs_preference(holdout), "bucket": "hour" if holdout else "day",
                                      "error_field": "http_status" if holdout else "level"}, indent=2) + "\n"
    expected = []
    for (bucket, service), rows in sorted(groups.items()):
        n = len(rows)
        expected.append({"bucket": bucket, "service": service, "count": n,
                         "errors": sum(failed for _, failed in rows),
                         "p95_ms": sorted(duration for duration, _ in rows)[n - n // 20 - 1]})
    error_rule = "integer http_status from 100 to 599, counted as an error at 500 or above" if holdout else (
        "level string INFO or ERROR, counted as an error for ERROR"
    )
    bucket_label = "UTC hour formatted YYYY-MM-DDTHH:00Z" if holdout else "UTC day formatted YYYY-MM-DD"
    rules = f"""Read `/app/inputs/part-*.jsonl` in filename order and line order within each file.
Skip malformed JSON and rows without a nonempty string id, nonempty string service, timezone-aware
ISO 8601 ts, a nonnegative integer duration_ms, and {error_rule}.
After filtering, keep the {inputs_preference(holdout)} valid occurrence of each id across all files.
Group by {bucket_label} and service. Count events and errors in each group.
For p95_ms use nearest rank: sort durations ascending and take the value at 1-based index
`ceil(0.95 * count)`. Write `/app/log_summary.json` as a JSON array with keys
`bucket`, `service`, `count`, `errors`, and `p95_ms`. Sort objects by bucket, then service."""
    wrong = [dict(expected[0], count=expected[0]["count"] + 1), *expected[1:]]
    return Case(shards, rules, {"file": "log_summary.json", "kind": "json", "rows": expected},
                [wrong, expected[:-1]], LOG_REFERENCE,
                "Produce the incident team's offline service summary from rotated application exports.")


LOG_REFERENCE = """\
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

app = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
folder = app / 'inputs'
settings = json.loads((folder / 'rules.json').read_text())
field = settings['error_field']
events = {}
for shard in sorted(folder.glob('part-*.jsonl')):
    for line in shard.read_text().splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            key, service = row['id'], row['service']
            if not all(isinstance(x, str) and x for x in (key, service)):
                continue
            stamp = datetime.fromisoformat(row['ts'].replace('Z', '+00:00'))
            duration, value = row['duration_ms'], row[field]
            if stamp.utcoffset() is None or type(duration) is not int or duration < 0:
                continue
            if field == 'http_status':
                if type(value) is not int or not 100 <= value <= 599:
                    continue
                failed = value >= 500
            else:
                if value not in ('INFO', 'ERROR'):
                    continue
                failed = value == 'ERROR'
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if settings['keep'] == 'first' and key in events:
            continue
        label = '%Y-%m-%dT%H:00Z' if settings['bucket'] == 'hour' else '%Y-%m-%d'
        group = stamp.astimezone(timezone.utc).strftime(label), service
        events[key] = group, duration, failed
groups = defaultdict(list)
for group, duration, failed in events.values():
    groups[group].append((duration, failed))
results = []
for (bucket, service), events in sorted(groups.items()):
    events.sort()
    count = len(events)
    results.append({'bucket': bucket, 'service': service, 'count': count,
                    'errors': sum(error for _, error in events),
                    'p95_ms': events[(95 * count + 99) // 100 - 1][0]})
(app / 'log_summary.json').write_text(json.dumps(results))
"""

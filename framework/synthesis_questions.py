"""Propose verifiable questions over a supplied or generated document collection."""

import itertools
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from synthesis_reasoning import json_case
from synthesis_reasoning_audit import self_questioning as legacy_audit

FIELDS = ("team", "topic", "stage")
MODES = ("unique", "earliest", "latest")


def documents(split, rng):
    teams, topics = (("support sales design", "intake billing returns") if split == "train"
                     else ("east north west", "inspection dispatch repairs"))
    records = []
    for team, topic, stage in itertools.product(teams.split(), topics.split(), ("draft", "final")):
        records.append({"id": f"D-{rng.getrandbits(64):016x}", "team": team, "topic": topic,
                        "stage": stage, "ordinal": len(records), "owner": f"O-{rng.getrandbits(48):012x}"})
    order = rng.sample(range(1000), len(records))
    for record, rank in zip(records, order):
        record["ordinal"] = rank
    rng.shuffle(records)
    return records


def resolve(rows, filters, mode):
    matches = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    if mode == "unique":
        return matches
    if len(matches) < 2:
        return []
    rank = (min if mode == "earliest" else max)(r["ordinal"] for r in matches)
    return [r for r in matches if r["ordinal"] == rank]


def validate(rows, proposal):
    mode, filters = proposal["mode"], proposal["filters"]
    if mode not in MODES or not isinstance(filters, dict) or not 2 <= len(filters) <= len(FIELDS):
        raise ValueError("Use a supported mode and at least two conditions")
    if any(k not in FIELDS or not isinstance(v, str) for k, v in filters.items()):
        raise ValueError("Unknown document condition")
    selected = resolve(rows, filters, mode)
    if len(selected) != 1:
        raise ValueError("Query needs exactly one winner and ranked queries need competitors")
    for field in filters:
        reduced = {k: v for k, v in filters.items() if k != field}
        if resolve(rows, reduced, mode) == selected:
            raise ValueError(f"Redundant condition: {field}")
    return selected[0]


def offline_proposal(rows, mode, rng):
    sizes = [3, 2] if mode == "unique" else [2, 3]
    options = [(r, fields) for r in rows for size in sizes for fields in itertools.combinations(FIELDS, size)]
    rng.shuffle(options)
    for row, keys in options:
        proposal = {"mode": mode, "filters": {k: row[k] for k in keys}}
        try:
            validate(rows, proposal)
        except ValueError:
            continue
        return proposal
    raise ValueError(f"Materials do not support a nonredundant {mode} question")


def model_proposal(rows, mode, topic, config, seed):
    url = config["base_url"].rstrip("/")
    loc = urlsplit(url)
    if loc.scheme != "https" or not loc.hostname or loc.username or loc.password or loc.query or loc.fragment:
        raise ValueError("Configure proposer.base_url as an HTTPS API root")
    key = os.environ.get(config["api_key_env"])
    if not key:
        raise ValueError(f"Set {config['api_key_env']} to use the question proposer")
    messages = [
        {"role": "system", "content": "Propose a structured question over the supplied records. Return "
         "one JSON object with mode, filters (at least two of team/topic/stage) and evidence_id (winning "
         "record). The mode must match the request. Unique means one match. Earliest/latest require "
         "at least two matches and a unique min/max ordinal. Dropping any one filter must change "
         "the winner or make it ambiguous. The framework renders the verified query as a question."},
        {"role": "user", "content": json.dumps({"mode": mode, "topic": topic, "sample": seed, "records": rows})}]
    for _ in range(3):
        payload = {"model": config["model"], "messages": messages, "temperature": 0.5}
        req = Request(url + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urlopen(req, timeout=90) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        try:
            obj = json.loads(content)
            if obj["mode"] != mode or validate(rows, obj)["id"] != obj["evidence_id"]:
                raise ValueError("Wrong mode or unsupported evidence")
            return {"mode": mode, "filters": obj["filters"]}
        except (ValueError, TypeError, KeyError) as error:
            messages.extend([{"role": "assistant", "content": content},
                             {"role": "user", "content": f"Revise the query. {error}"}])
    raise ValueError("Proposer failed independent evidence checks after three attempts")


def self_questioning(split, index, seed, *, materials_file=None, topic="document ownership",
                    focus=None, proposer=None, replay_manifest=None):
    rng = random.Random(seed)
    rows = (json.loads(Path(materials_file).expanduser().read_text())[split]
            if materials_file else documents(split, rng))
    if not rows or any(not isinstance(row, dict) or type(row.get("ordinal")) is not int
                       or not all(isinstance(row.get(k), str) and row[k] for k in (*FIELDS, "id", "owner"))
                       for row in rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Each document needs a unique id, owner, team, topic, stage and integer ordinal")
    mode = focus or MODES[(index - 1) % len(MODES)]
    if mode not in MODES:
        raise ValueError(f"Unknown question focus {mode}")
    if replay_manifest and proposer:
        raise ValueError("Select either a saved proposal or a live proposer")
    if replay_manifest:
        manifest = json.loads(Path(replay_manifest).expanduser().read_text())
        proposals = [task["generation"]["proposal"] for task in manifest["tasks"]
                     if task["split"] == split and task["seed"] == seed
                     and "proposal" in task.get("generation", {})]
        if len(proposals) != 1 or proposals[0]["mode"] != mode:
            raise ValueError("Saved manifest has no unique matching proposal for this seed and mode")
        p = proposals[0]
        source = "replay"
    elif proposer:
        p = model_proposal(rows, mode, topic, proposer, seed)
        source = proposer["model"]
    else:
        p = offline_proposal(rows, mode, rng)
        source = "offline"
    row = validate(rows, p)
    criteria = " and ".join(f"{k} equals {json.dumps(v)}" for k, v in sorted(p["filters"].items()))
    target = {"unique": "the unique matching document", "earliest": "the matching document with the lowest ordinal",
              "latest": "the matching document with the highest ordinal"}[mode]
    question = f"Among the documents where {criteria}, who owns {target}?"
    rules = (question + "\nUse /app/inputs/documents.json and write /app/answer.json with record_id "
             "and owner. The ordinal is the document's chronological sequence number.")
    answer = {"record_id": row["id"], "owner": row["owner"]}
    other = next(r for r in rows if r["id"] != row["id"])
    case = json_case({"documents.json": rows}, rules, answer,
                     [dict(answer, record_id=other["id"]), {"record_id": row["id"]}],
                    REFERENCE.replace("__PROPOSAL__", repr(p)), topic, "question-" + mode)
    case.provenance = {"proposer": source, "proposal": p}
    return case


REFERENCE = """import json
from pathlib import Path
import sys
app = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
query = __PROPOSAL__
rows = json.loads((app / 'inputs/documents.json').read_text())
hits = [r for r in rows if all(r[k] == v for k,v in query['filters'].items())]
hits.sort(key=lambda r:r['ordinal'])
record = hits[-1] if query['mode'] == 'latest' else hits[0]
(app / 'answer.json').write_text(json.dumps({'record_id':record['id'], 'owner':record['owner']}))
"""


def audit(inputs):
    if (inputs / "records.json").exists():
        return legacy_audit(inputs)
    # Recover the actual question from the *public* text, independently of the proposer's private query.
    text = (inputs.parents[1] / "instruction.md").read_text()
    match = re.search(r'^Among the documents where (.+), who owns (.+)\?$', text, re.MULTILINE)
    if not match:
        raise ValueError("Missing document question in instruction")
    decoder = json.JSONDecoder()
    criteria, target = match.groups()
    filters = {}
    while criteria:
        field, sep, literal = criteria.partition(" equals ")
        if not sep or field not in FIELDS or field in filters:
            raise ValueError("Invalid condition in public question")
        value, end = decoder.raw_decode(literal)
        filters[field] = value
        criteria = literal[end:]
        if criteria and not criteria.startswith(" and "):
            raise ValueError("Invalid conjunction in public question")
        criteria = criteria[5:]
    ops = {"the unique matching document": None, "the matching document with the lowest ordinal": "MIN",
           "the matching document with the highest ordinal": "MAX"}
    aggregate = ops[target]
    docs = json.loads((inputs / "documents.json").read_text())
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE docs (id,owner,team,topic,stage,ordinal INTEGER)")
        db.executemany("INSERT INTO docs VALUES (:id,:owner,:team,:topic,:stage,:ordinal)", docs)
        sql = "WITH hits AS (SELECT * FROM docs WHERE " + " AND ".join(f"{k}=:{k}" for k in filters) + ") "
        sql += "SELECT id,owner FROM hits"
        if aggregate:
            sql += f" WHERE ordinal=(SELECT {aggregate}(ordinal) FROM hits) AND (SELECT count(*) FROM hits)>1"
        result = db.execute(sql, filters).fetchall()
    if len(result) != 1:
        raise ValueError("Public document question lacks a unique answer")
    return dict(zip(("record_id", "owner"), result[0]))

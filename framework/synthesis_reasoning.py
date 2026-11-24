"""Seeded tool-composition, question-generation, and evidence-chain cases."""

from collections import Counter
import json
from pathlib import Path
import random
import re

from synthesis_methods import Case


def json_text(value):
    return json.dumps(value, indent=2) + "\n"


def json_case(files, rules, answer, wrong, reference, scenario, environment):
    return Case(
        {name: json_text(value) for name, value in files.items()},
        rules, {"file": "answer.json", "kind": "json", "rows": answer},
        wrong, reference, scenario, environment,
    )


REFERENCE_HEADER = """\
import json
from pathlib import Path
import sys

app = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
inputs = app / 'inputs'
request = json.loads((inputs / 'request.json').read_text())
"""


TOOL_GRAPHS = {
    "train": {
        "lookup_item": {"located in": "lookup_bin"},
        "lookup_bin": {"belongs to": "lookup_warehouse"},
        "lookup_warehouse": {"dispatches with": "lookup_carrier", "covers": "lookup_region"},
        "lookup_carrier": {"assigned to": "lookup_service_desk"},
        "lookup_region": {},
        "lookup_service_desk": {},
    },
    "holdout": {
        "lookup_exhibit": {"stored in": "lookup_collection"},
        "lookup_collection": {"managed by": "lookup_department"},
        "lookup_department": {"uses": "lookup_conservation_lab", "reports to": "lookup_campus"},
        "lookup_conservation_lab": {"supported by": "lookup_vendor"},
        "lookup_campus": {},
        "lookup_vendor": {"assigned to": "lookup_contract_office"},
        "lookup_contract_office": {},
    },
}


def routes_from(graph, start, path=(), visited=frozenset()):
    visited = visited | {start}
    for relation, target in graph[start].items():
        if target in visited:
            raise ValueError("Tool dependency graph must be acyclic")
        route = (*path, (relation, target))
        if graph[target]:
            yield from routes_from(graph, target, route, visited)
        else:
            yield route


def toolspace(split, index, seed, *, graph_file=None):
    rng = random.Random(seed)
    graphs = TOOL_GRAPHS if graph_file is None else json.loads(Path(graph_file).expanduser().read_text())
    graph = graphs[split]
    if not isinstance(graph, dict) or not graph or not all(
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", tool) and isinstance(links, dict)
        and all(relation.strip() and isinstance(target, str) and target in graph
                for relation, target in links.items())
        for tool, links in graph.items()
    ):
        raise ValueError(f"Invalid tool dependency graph for {split}")
    start = next(iter(graph))
    routes = list(routes_from(graph, start))
    if {start, *(tool for route in routes for _, tool in route)} != graph.keys():
        raise ValueError(f"All tools in the {split} graph must be reachable from its first tool")
    destinations = Counter(path[-1][1] for path in routes)
    usable = [path for path in routes if len(path) >= 3 and destinations[path[-1][1]] == 1]
    if not usable:
        raise ValueError(f"Tool graph for {split} needs an unambiguous destination at least three links away")
    route = rng.choice(usable)
    ids = {
        tool: [f"R-{number:08x}" for number in rng.sample(range(16**8), 12)]
        for tool in graph
    }
    catalog = {tool: {"entity": tool.removeprefix("lookup_").replace("_", " "),
                      "links": links} for tool, links in graph.items()}
    records = {}
    for tool, links in graph.items():
        targets = {relation: rng.sample(ids[target], len(ids[target])) for relation, target in links.items()}
        records[tool] = {
            identifier: {"identifier": identifier, "label": f"Label-{rng.getrandbits(48):012x}",
                         "links": {relation: values[offset] for relation, values in targets.items()}}
            for offset, identifier in enumerate(ids[tool])
        }

    start_id = rng.choice(ids[start])
    goal = route[-1][1]
    path = [start_id]
    tool = start
    for relation, target in route:
        path.append(records[tool][path[-1]]["links"][relation])
        tool = target
    final = records[goal][path[-1]]
    answer = {"record_ids": path, "label": final["label"]}
    case = json_case(
        {"request.json": {"start_tool": start, "start_id": start_id, "goal_tool": goal},
         "tool-data.json": {"catalog": catalog, "records": records}},
        f"""Find the {catalog[goal]["entity"]} associated with the starting
{catalog[start]["entity"]} in `/app/inputs/request.json`.
Run `python3 /app/inputs/tools.py` to list the catalog and each record's available links.
Run `python3 /app/inputs/tools.py TOOL IDENTIFIER` to retrieve a record.
Each link returns the identifier to pass to the destination tool. Follow links from
the starting record until you reach the requested entity.
Write `/app/answer.json` as an object with `record_ids` (the identifiers visited,
including the start and destination in order) and `label` (the destination's label).""",
        answer, [dict(answer, record_ids=path[1:]), dict(answer, label=records[start][start_id]["label"])],
        TOOL_REFERENCE, "Trace the catalog entry through several linked records.", "tool-catalog",
    )
    case.files["tools.py"] = Path(__file__).with_name("synthesis_tool_cli.py").read_text()
    return case


TOOL_REFERENCE = REFERENCE_HEADER + """\
import subprocess

def call(*args):
    return json.loads(subprocess.check_output(
        [sys.executable, str(inputs / 'tools.py'), *args], text=True))

catalog = call()
def find(tool, identifier, path):
    row = call(tool, identifier)
    path = [*path, identifier]
    if tool == request['goal_tool']:
        return {'record_ids': path, 'label': row['label']}
    for relation, target in catalog[tool]['links'].items():
        answer = find(target, row['links'][relation], path)
        if answer is not None:
            return answer
    return None

answer = find(request['start_tool'], request['start_id'], [])
(app / 'answer.json').write_text(json.dumps(answer))
"""


QUESTION_SCHEMAS = {
    "train": {
        "fields": {"team": ("design", "sales", "support", "operations"),
                   "quarter": ("Q1", "Q2", "Q3", "Q4"),
                   "format": ("memo", "brief", "checklist", "slides")},
        "answers": ("owner_code", "archive_code"),
        "subject": "documents",
    },
    "holdout": {
        "fields": {"site": ("north", "east", "south", "west"),
                   "class": ("intro", "practical", "review", "advanced"),
                   "session": ("morning", "noon", "afternoon", "evening")},
        "answers": ("room_code", "facilitator_code"),
        "subject": "workshops",
    },
}


def self_questioning(split, index, seed):
    rng = random.Random(seed)
    schema = QUESTION_SCHEMAS[split]
    fields = schema["fields"]
    rows = [
        {"id": f"Q-{number:03}",
         **{field: rng.choice(values) for field, values in fields.items()},
         **{answer: f"A-{rng.getrandbits(48):012x}" for answer in schema["answers"]}}
        for number in range(24)
    ]
    # Propose questions from observations, then answer them against the entire source.
    # Keep only proposals whose full filter resolves uniquely and whose individual clues do not.
    candidates = []
    for row in rows:
        conditions = {field: row[field] for field in fields}
        matches = [other for other in rows if all(other[key] == value for key, value in conditions.items())]
        if len(matches) == 1 and all(
            sum(other[key] == value for other in rows) > 1 for key, value in conditions.items()
        ):
            candidates.append((conditions, matches[0]))
    if not candidates:
        # A fixed proposal with controlled distractors guarantees coverage for every seed.
        conditions = {field: values[0] for field, values in fields.items()}
        for row in rows:
            row.update({field: values[1] for field, values in fields.items()})
        rows[0].update(conditions)
        for offset, (field, value) in enumerate(conditions.items(), 1):
            rows[offset][field] = value
        candidates.append((conditions, rows[0]))
    conditions, match = rng.choice(candidates)
    target = rng.choice(schema["answers"])
    wrong_match = next(row for row in rows if row["id"] != match["id"] and
                       any(row[key] == value for key, value in conditions.items()))
    clues = ", ".join(f"{key} = {value}" for key, value in conditions.items())
    return json_case(
        {"records.json": rows, "request.json": {"conditions": conditions, "field": target}},
        f"""What is the `{target}` for the record matching all of these conditions: {clues}?
Use `/app/inputs/records.json`. Write `/app/answer.json` as an object with
`record_id` (the matched row's `id`) and `value` (its `{target}`).""",
        {"record_id": match["id"], "value": match[target]},
        [{"record_id": match["id"]}, {"record_id": wrong_match["id"], "value": wrong_match[target]}],
        QUESTION_REFERENCE, f"Answer a question about a catalog of internal {schema['subject']}.", "record-catalog",
    )


QUESTION_REFERENCE = REFERENCE_HEADER + """\
rows = json.loads((inputs / 'records.json').read_text())
row, = (record for record in rows
        if all(record[key] == value for key, value in request['conditions'].items()))
answer = {'record_id': row['id'], 'value': row[request['field']]}
(app / 'answer.json').write_text(json.dumps(answer))
"""


HOP_RELATIONS = {
    "train": ("uses-component", "supplied-by", "has-contract", "managed-by"),
    "holdout": ("belongs-to-program", "assigned-to-office", "uses-budget", "funded-by", "reviewed-by", "stored-at"),
}


def multi_hop(split, index, seed):
    rng = random.Random(seed)
    available = HOP_RELATIONS[split]
    # Use 2–4 joins for training and 4–6 joins on a different relation schema for holdout.
    relations = available[:len(available) - 2 + (index - 1) % 3]
    layers = [
        [f"E-{value:08x}" for value in rng.sample(range(16**8), 7)]
        for _ in range(len(relations) + 1)
    ]
    # Namespace entity IDs by layer to rule out accidental shortcuts.
    layers = [[f"{depth}-{entity}" for entity in layer] for depth, layer in enumerate(layers)]
    edges = []
    for depth, relation in enumerate(relations):
        targets = rng.sample(layers[depth + 1], len(layers[depth + 1]))
        for source, target in zip(layers[depth], targets):
            edges.append({"from": source, "relation": relation, "to": target})
            decoy = rng.choice([entity for entity in targets if entity != target])
            edges.append({"from": source, "relation": "mentioned-with", "to": decoy})

    starts = rng.sample(layers[0], 3)
    by_source = {(edge["from"], edge["relation"]): edge["to"] for edge in edges}
    answer = []
    for start in starts:
        path = [start]
        for relation in relations:
            path.append(by_source[path[-1], relation])
        answer.append({"start": start, "path": path})

    rng.shuffle(edges)
    files = {f"facts-{n + 1}.json": edges[n::3] for n in range(3)}
    files["request.json"] = {"starts": starts, "relations": relations}
    wrong = [dict(answer[0], path=answer[0]["path"][:-1]), *answer[1:]]
    return json_case(
        files,
        """For each starting entity in `/app/inputs/request.json`, resolve the listed
relations in order using the directed facts in `/app/inputs/facts-*.json`.
Each fact carries `from`, `relation` and `to`. Join each answer to the next fact's
`from` using the exact relation for that hop. Write `/app/answer.json` as an array
of objects with `start` and `path`. Include the start and all subsequent entity
IDs in the path and preserve the order of the starting entities.""",
        answer, [wrong, answer[1:], list(reversed(answer))], HOP_REFERENCE,
        "Link records across several independently exported relationship files.", "fact-shards",
    )


HOP_REFERENCE = REFERENCE_HEADER + """\
facts = [row for file in inputs.glob('facts-*.json') for row in json.loads(file.read_text())]
lookup = {(fact['from'], fact['relation']): fact['to'] for fact in facts}
result = []
for start in request['starts']:
    path = [start]
    for relation in request['relations']:
        path.append(lookup[path[-1], relation])
    result.append({'start': start, 'path': path})
(app / 'answer.json').write_text(json.dumps(result))
"""

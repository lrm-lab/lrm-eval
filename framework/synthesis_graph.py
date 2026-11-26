"""Generate paired evidence-graph questions; a single fact changes across each pair."""

import json
from pathlib import Path
import random
import sqlite3
import tomllib

from synthesis_reasoning import json_case
from synthesis_reasoning_audit import multi_hop as legacy_audit

DOMAINS = {
    "train": ("incident", "component", "supplier", "support contact"),
    "holdout": ("review", "program", "assessor", "review coordinator"),
}

QUESTIONS = {
    "bridge": "Who is the {contact} for the {provider} responsible for this {subject}?",
    "intersection": "Who is the {contact} for the {provider} responsible for this {subject} "
                    "and accredited for the filing region?",
    "temporal": "Who was the {contact} for the {provider} responsible for this {subject} "
                "on the filing date? Use the most recent assignment effective by that date.",
}


def multi_hop(split, index, seed):
    rng = random.Random(seed)
    kind = tuple(QUESTIONS)[((index - 1) // 2) % len(QUESTIONS)]
    amended = index % 2 == 0
    subject, asset, provider, contact = DOMAINS[split]
    def ids(prefix, size):
        return [f"{prefix}-{value:05x}" for value in rng.sample(range(16**5), size)]
    parts, suppliers, contacts = ids(asset, 4), ids(provider, 5), ids("desk", 5)
    regions = ids("region", 3)
    request = {"case": ids(subject, 1)[0], "region": regions[0], "date": "2025-06-10"}
    source = [{"id": uid, "case": case, "item": part} for uid, case, part in zip(
        ids("source", 4), [request["case"], *ids(subject + "-other", 3)], parts)]
    assignments = [{"id": uid, "item": part, "provider": firm, "effective": "2025-01-01"}
                   for uid, part, firm in zip(ids("assignment", 4), parts, suppliers)]
    directory = [{"id": uid, "provider": firm, "contact": desk}
                 for uid, firm, desk in zip(ids("entry", 5), suppliers, contacts)]
    permits = [{"id": uid, "provider": firm, "region": region}
               for uid, firm, region in zip(ids("permit", 5), suppliers, [regions[1], *([regions[0]] * 4)])]
    if kind == "bridge":
        source[0]["item"] = parts[1] if amended else parts[0]
    else:
        assignments.append({"id": ids("renewal", 1)[0], "item": parts[0], "provider": suppliers[4],
                            "effective": "2025-06-09" if amended else "2025-06-11"})
        if kind == "intersection":
            # Both assignments are candidates. Swapping the filing region selects exactly one permit.
            assignments[-1]["effective"] = "2025-01-01"
            request["region"] = regions[1] if amended else regions[0]
    chosen = 1 if kind == "bridge" and amended else 0
    if (kind == "temporal" and amended) or (kind == "intersection" and not amended):
        chosen = 4
    evidence = [source[0]["id"], assignments[chosen]["id"], directory[chosen]["id"]]
    if kind == "intersection":
        evidence.append(permits[chosen]["id"])
    answer = {"contact": contacts[chosen], "evidence": sorted(evidence)}
    files = {"case.json": request, "sources.json": source, "assignments.json": assignments,
             "directory.json": directory, "permits.json": permits}
    for rows in (source, assignments, directory, permits):
        rng.shuffle(rows)
    question = QUESTIONS[kind].format(subject=subject, provider=provider, contact=contact)
    rules = (question + "\nThe filing is in /app/inputs/case.json. The other JSON files are "
             "source records. Submit /app/answer.json with contact (the desk ID) and evidence "
             "(all supporting source record IDs, sorted lexicographically).")
    alternate = 1 - chosen if kind == "bridge" else 4 - chosen
    case = json_case(files, rules, answer, [dict(answer, evidence=evidence[:1]),
                    dict(answer, contact=contacts[alternate])], REFERENCE.replace("__KIND__", repr(kind)),
                    f"Resolve the {subject} from the {asset} and {provider} records.", "evidence-" + kind)
    return case


REFERENCE = """import json
from pathlib import Path
import sys
app = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app')
def read(name):
    return json.loads((app / 'inputs' / (name + '.json')).read_text())
case = read('case')
source, = [r for r in read('sources') if r['case'] == case['case']]
assignments = [r for r in read('assignments') if r['item'] == source['item']]
evidence = [source['id']]
kind = __KIND__
if kind == 'intersection':
    permits = {r['provider']:r['id'] for r in read('permits') if r['region'] == case['region']}
    assignment, = [r for r in assignments if r['provider'] in permits]
    evidence.append(permits[assignment['provider']])
elif kind == 'temporal':
    assignment = max((r for r in assignments if r['effective'] <= case['date']), key=lambda r:r['effective'])
else:
    assignment, = assignments
entry, = [r for r in read('directory') if r['provider'] == assignment['provider']]
evidence.extend([assignment['id'],entry['id']])
(app / 'answer.json').write_text(json.dumps({'contact':entry['contact'], 'evidence':sorted(evidence)}))
"""


def audit(inputs):
    if (inputs / "request.json").exists():
        return legacy_audit(inputs)
    meta = tomllib.loads((inputs.parents[1] / "task.toml").read_text())["metadata"]
    kind = meta["environment"].removeprefix("evidence-")
    case = json.loads((inputs / "case.json").read_text())
    with sqlite3.connect(":memory:") as db:
        for table, fields in {"sources": "id,subject,item", "assignments": "id,item,provider,effective",
                              "directory": "id,provider,contact", "permits": "id,provider,region"}.items():
            columns = fields.split(",")
            db.execute(f"CREATE TABLE {table} ({fields})")
            rows = json.loads((inputs / (table + ".json")).read_text())
            db.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for c in columns)})",
                           [[r["case" if c == "subject" else c] for c in columns] for r in rows])
        select = "SELECT d.contact,s.id,a.id,d.id"
        joins = " FROM sources s JOIN assignments a USING(item) JOIN directory d USING(provider)"
        where = " WHERE s.subject=:case"
        if kind == "intersection":
            select += ",p.id"
            joins += " JOIN permits p USING(provider)"
            where += " AND p.region=:region"
        elif kind == "temporal":
            where += """ AND a.effective=(SELECT MAX(b.effective) FROM assignments b
                        WHERE b.item=s.item AND b.effective<=:date)"""
        elif kind != "bridge":
            raise ValueError(f"Unknown evidence query {kind}")
        query = select + joins + where
        rows = db.execute(query, case).fetchall()
        if len(rows) != 1:
            raise ValueError("Evidence query must yield exactly one supported answer")
        contact, *evidence = rows[0]
        tables = ["sources", "assignments", "directory"]
        if kind == "intersection":
            tables.append("permits")
        for table, record_id in zip(tables, evidence):
            db.execute("SAVEPOINT omit_fact")
            db.execute(f"DELETE FROM {table} WHERE id=?", (record_id,))
            remaining = db.execute(query, case).fetchall()
            if any(row[0] == contact for row in remaining):
                raise ValueError("A reported supporting fact is unnecessary")
            db.execute("ROLLBACK TO omit_fact")
            db.execute("RELEASE omit_fact")
        return {"contact": contact, "evidence": sorted(evidence)}

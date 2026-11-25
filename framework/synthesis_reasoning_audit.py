"""Independent checks of the generated tools, record questions, and evidence paths."""

import json


def read(inputs, name):
    return json.loads((inputs / name).read_text())


def toolspace(inputs):
    request = read(inputs, "request.json")
    data = read(inputs, "tool-data.json")
    return tool_answer(request, data)


def tool_answer(request, data):
    queue = [(request["start_tool"], [request["start_id"]], frozenset())]
    answers = []
    while queue:
        tool, path, visited = queue.pop()
        if tool in visited:
            raise ValueError("Tool graph contains a cycle")
        row = data["records"][tool][path[-1]]
        if tool == request["goal_tool"]:
            answers.append({"record_ids": path, "label": row["label"]})
            continue
        for link, identifier in row["links"].items():
            target = data["catalog"][tool]["links"][link]
            queue.append((target, [*path, identifier], visited | {tool}))
    if len(answers) != 1 or len(answers[0]["record_ids"]) < 4:
        raise ValueError("Tool request must have one answer requiring at least three links")
    return answers[0]


def self_questioning(inputs):
    request = read(inputs, "request.json")
    rows = read(inputs, "records.json")
    matching_ids = set(range(len(rows)))
    for key, value in request["conditions"].items():
        matching = {index for index, record in enumerate(rows) if record.get(key) == value}
        if len(matching) < 2:
            raise ValueError("Question has an individually identifying clue")
        matching_ids &= matching
    if len(matching_ids) != 1:
        raise ValueError("Question does not identify one record")
    record = rows[matching_ids.pop()]
    return {"record_id": record["id"], "value": record[request["field"]]}


def multi_hop(inputs):
    request = read(inputs, "request.json")
    facts = [(row["from"], row["relation"], row["to"])
             for file in inputs.glob("facts-*.json") for row in read(inputs, file.name)]
    paths = [[start] for start in request["starts"]]
    for relation in request["relations"]:
        for path in paths:
            matches = [target for source, label, target in facts if source == path[-1] and label == relation]
            if len(matches) != 1 or matches[0] in path:
                raise ValueError("Fact chain must have one acyclic continuation")
            path.append(matches[0])
    return [{"start": path[0], "path": path} for path in paths]

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1] / "framework"
sys.path.insert(0, str(FRAMEWORK))

import synthesis_questions
from synthesis_tool_service import Catalog


class IndependentMethods(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = {**os.environ, "PYTHONPATH": str(self.root) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def generate(self, name, methods, count=1, config=None):
        call = [sys.executable, "-B", str(FRAMEWORK / "synthesize.py"), "generate", name,
                "--train-per-method", str(count), "--holdout-per-method", "0", "--seed", "42",
                "--output", str(self.root)]
        if config:
            call += ["--config", str(config)]
        for method in methods:
            call += ["--method", method]
        subprocess.run(call, check=True, capture_output=True, env=self.env)
        return self.root / name

    def check(self, package, config=None):
        call = [sys.executable, "-B", str(FRAMEWORK / "synthesize.py"), "check", str(package)]
        if config:
            call += ["--config", str(config)]
        result = subprocess.run(call, check=True, capture_output=True, env=self.env)
        return json.loads(result.stdout)

    def test_single_and_mixed_calls_and_extension(self):
        config = self.root / "team.toml"
        (self.root / "team_methods.py").write_text(
            "from synthesis_methods import Case\n"
            "def square(split, index, seed):\n"
            "    n = seed % 1000\n"
            "    reference = \"import sys, pathlib; app = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 "
            "else pathlib.Path('/app'); (app/'answer.json').write_text(str(int((app/'inputs/n').read_text())**2))\"\n"
            "    return Case({'n':str(n)}, 'Square inputs/n into answer.json.', "
            "{'file':'answer.json','kind':'json','rows':n*n}, [-1], reference, 'Compute a square.')\n"
        )
        contents = (FRAMEWORK / "harbor.toml").read_text()
        config.write_text(contents.replace("[synthesis.methods]", '[synthesis.methods]\n'
                          'team-questions = "synthesis_questions:self_questioning"\n'
                          'team-square = "team_methods:square"')
                          + '\n[synthesis.options.team-questions]\nfocus = "latest"\n')
        methods = ["multi-hop", "toolspace", "self-questioning", "team-questions", "team-square"]
        combined = self.generate("mixed", methods, config=config)
        self.assertEqual(self.check(combined, config)["passed"], 5)
        for method in methods:
            single = self.generate("only-" + method, [method], config=config)
            relative = Path("train") / (method + "-train-001")
            def files(path):
                return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
            self.assertEqual(files(single / relative), files(combined / relative))

    def test_graph_pairs_change_one_fact_and_require_it(self):
        bundle = self.generate("pairs", ["multi-hop"], count=6)
        def scalars(value):
            if isinstance(value, dict):
                for k in sorted(value):
                    yield from scalars(value[k])
            elif isinstance(value, list):
                for item in value:
                    yield from scalars(item)
            else:
                yield value
        tasks = sorted((bundle / "train").iterdir())
        for first, second in zip(tasks[::2], tasks[1::2]):
            def facts(task):
                return list(scalars([json.loads(p.read_text())
                            for p in sorted((task / "environment" / "inputs").glob("*.json"))]))
            left, right = facts(first), facts(second)
            self.assertEqual(len(left), len(right))
            self.assertEqual(sum(a != b for a, b in zip(left, right)), 1)
            def answer(task):
                return json.loads((task / "tests/expected.json").read_text())["rows"]["contact"]
            self.assertNotEqual(answer(first), answer(second))
        self.assertEqual(self.check(bundle)["independent_audit"], 6)

    def test_tool_requires_observations_and_recovers(self):
        bundle = self.generate("tools", ["toolspace"], count=3)
        task = bundle / "train/toolspace-train-003"
        env = task / "environment"
        self.assertEqual({p.name for p in (env / "inputs").iterdir()}, {"request.json", "tools.py"})
        data = json.loads((env / "catalog/fixture.json").read_text())
        goal = json.loads((task / "tests/expected.json").read_text())["rows"]
        api = Catalog(data)
        path = goal["record_ids"]
        with self.assertRaises(ValueError):
            api.call(["prepare", path[-1]])
        tool = data["start_tool"]
        for index, identifier in enumerate(path):
            row = api.call([tool, identifier])
            if identifier != path[-1]:
                edge, = (k for k,v in row["links"].items() if v == path[index + 1])
                tool = data["catalog"][tool]["links"][edge]
        initial = api.call(["prepare", path[-1]])
        action = [initial["action"], path[-1], str(initial["version"])]
        with self.assertRaises(ValueError):
            api.call(action)
        action.append(api.call(["authorize", *action[1:]])["authorization"])
        self.assertEqual(api.call(action)["error"], "stale_version")
        self.assertIsNone(api.call(["status"])["completed"])
        fresh = api.call(["prepare", path[-1]])
        action[2] = str(fresh["version"])
        with self.assertRaises(ValueError):
            api.call(action)
        action[3] = api.call(["authorize", *action[1:3]])["authorization"]
        self.assertEqual(api.call(action)["receipt"], goal["receipt"])
        self.assertEqual(api.call(action), api.call(["status"])["completed"])

    def test_question_model_rejects_wrong_evidence_then_replays(self):
        import io
        import random
        rows = synthesis_questions.documents("train", random.Random(2))
        row = rows[0]
        proposal = {"mode": "unique", "filters": {k: row[k] for k in synthesis_questions.FIELDS}}
        def reply(witness):
            text = json.dumps(dict(proposal, evidence_id=witness))
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}}]}).encode())
        config = {"base_url": "https://example.test/v1", "model": "test", "api_key_env": "SYNTH_TEST_KEY"}
        with patch.dict("os.environ", {"SYNTH_TEST_KEY": "test"}), patch.object(
                synthesis_questions, "urlopen", side_effect=[reply("missing"), reply(row["id"])]) as calls:
            self.assertEqual(synthesis_questions.model_proposal(rows, "unique", "owner", config, 4), proposal)
            self.assertEqual(calls.call_count, 2)
        bundle = self.generate("original", ["self-questioning"], count=3)
        replay_config = self.root / "replay.toml"
        replay_config.write_text((FRAMEWORK / "harbor.toml").read_text() +
                                 '\n[synthesis.options.self-questioning]\nreplay_manifest = ' +
                                 json.dumps(str(bundle / "synthesis.json")) + '\n')
        replay = self.generate("replay", ["self-questioning"], count=3, config=replay_config)
        for expected in (bundle / "train").glob("*/tests/expected.json"):
            self.assertEqual(expected.read_bytes(), (replay / expected.relative_to(bundle)).read_bytes())
        self.assertEqual(self.check(replay)["passed"], 3)


if __name__ == "__main__":
    unittest.main()

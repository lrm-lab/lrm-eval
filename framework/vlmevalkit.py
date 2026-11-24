"""Run VLMEvalKit's datasets and scorers in their own Python environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def options_for(suite, path):
    options = suite.get("vlmevalkit") or {}
    if not options.get("dataset") or not path.exists():
        raise ValueError("Set vlmevalkit.dataset and a readable TSV or LMUData directory")
    if path.is_file() and (path.suffix != ".tsv" or not options.get("dataset_class")):
        raise ValueError("Local image TSV inputs require vlmevalkit.dataset_class")
    if options.get("mode", "all") not in ("all", "infer"):
        raise ValueError("VLMEvalKit mode must be all or infer")
    for section in ("model_options", "judge"):
        if any(key in options.get(section, {}) for key in ("key", "api_key", "api_base")):
            raise ValueError("Pass VLMEvalKit gateway and judge credentials through environment variables")
    return options


def launch(args, suite, path, job, mode):
    options = options_for(suite, path)
    kit = Path(os.environ.get("VLMEVALKIT_ROOT", "")).expanduser().resolve()
    if not (kit / "vlmeval" / "__init__.py").is_file():
        raise ValueError("Set VLMEVALKIT_ROOT to the VLMEvalKit checkout")
    interpreter = os.environ.get("VLMEVALKIT_PYTHON", sys.executable)
    interpreter = shutil.which(os.path.expanduser(interpreter))
    if interpreter is None:
        raise ValueError("VLMEvalKit Python interpreter is unavailable")
    job.mkdir(parents=True, exist_ok=False)
    request = job / "vlmeval-request.json"
    write_json(request, {"options": {**options, "mode": mode}, "data": str(path), "model": args.model,
                         "base_url": args.base_url, "tasks": args.task,
                         "name": suite["name"], "concurrent": args.concurrent})
    # VLMEvalKit reads the checkout's .env on import and during judge setup.
    # Stage only the package so credentials and dataset caches stay run-scoped.
    package = job / "python" / "vlmeval"
    shutil.copytree(kit / "vlmeval", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env"))
    (package.parent / ".env").touch()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package.parent)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LMUData"] = str(job / "lmu-data")
    cache = job / "cache"
    env["XDG_CACHE_HOME"] = str(cache)
    env["HF_HOME"] = str(cache / "huggingface")
    (job / "lmu-data").mkdir()
    log = job / "vlmevalkit.log"
    with log.open("wb") as stream:
        code = subprocess.run([interpreter, "-P", "-B", str(Path(__file__).resolve()), str(request)],
                              cwd=job, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        print(f"VLMEvalKit exited with code {code}. Local log: {log}", file=sys.stderr)
    return code


def api_url(url):
    url = url.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def load_dataset(source, options):
    import vlmeval.dataset as datasets
    from vlmeval.dataset.image_base import ImageBaseDataset
    from vlmeval.smp import load

    name = options["dataset"]
    kwargs = options.get("dataset_options", {})
    if source.is_dir():
        shutil.copytree(source, os.environ["LMUData"], dirs_exist_ok=True)
        dataset = datasets.build_dataset(name, **kwargs)
        if dataset is None:
            raise ValueError(f"VLMEvalKit has no dataset {name}")
        return dataset
    cls = getattr(datasets, options["dataset_class"], None)
    if not isinstance(cls, type) or not issubclass(cls, ImageBaseDataset):
        raise ValueError("Local TSV input requires a native VLMEvalKit image dataset class")

    class LocalImageDataset(cls):
        def load_data(self, dataset):
            return load(str(source))

    dataset = LocalImageDataset(dataset=name, **kwargs)
    if not dataset.meta_only and "image_path" in dataset.data:
        root = Path(dataset.img_root).resolve()
        if not root.is_relative_to(Path(os.environ["LMUData"]).resolve()):
            raise ValueError("Native image output must stay inside the job cache")
        for entry in dataset.data["image_path"]:
            paths = entry if isinstance(entry, list) else [entry]
            if any(not (root / name).resolve().is_relative_to(root) for name in paths):
                raise ValueError("Embedded image paths must stay inside the job cache")
    return dataset


def nonempty_score(value):
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scores_from_details(folder, options):
    from vlmeval.smp import load

    file = options.get("detail_file")
    field = options.get("detail_score_key")
    if file is None and field is None:
        return {}
    if not file or not field or Path(file).name != file:
        raise ValueError("Set a native detail_file basename and detail_score_key")
    rows = load(str(folder / file))
    if not {"index", field}.issubset(rows.columns):
        raise ValueError("Native score details require index and the configured score column")
    if rows["index"].astype(str).duplicated().any():
        raise ValueError("Native score details have duplicate sample IDs")
    return {str(row["index"]): nonempty_score(row[field]) for _, row in rows.iterrows()}


def metric_result(metrics, key):
    records = metrics if isinstance(metrics, list) else [metrics]
    values = []
    for row in records:
        if isinstance(row, dict) and key in row and nonempty_score(row[key]) is not None:
            values.append(nonempty_score(row[key]))
    if len(values) != 1:
        raise ValueError(f"Expected one native reported score for {key}")
    return values[0]


def worker(path):
    from vlmeval.api.gpt import GPT4V
    from vlmeval.smp import dump
    from vlmeval.smp.file import INFER_FAIL_MSG

    job = path.parent
    request = json.loads(path.read_text())
    options = request["options"]
    dataset = load_dataset(Path(request["data"]), options)
    native = job / "vlmevalkit"
    native.mkdir()
    data = dataset.data
    if data["index"].astype(str).duplicated().any():
        raise ValueError("VLMEvalKit input has duplicate sample IDs")
    if request["tasks"]:
        missing = set(request["tasks"]) - set(data["index"].astype(str))
        if missing:
            raise ValueError(f"Unknown VLMEvalKit sample IDs: {', '.join(sorted(missing))}")
        data = data[data["index"].astype(str).isin(request["tasks"])]
    if not len(data):
        raise ValueError("No VLMEvalKit samples selected")
    write_json(job / "result.json", {"n_total_trials": len(data)})

    class RecordingVLM(GPT4V):
        prepared = None
        http_status = None

        def prepare_inputs(self, inputs, system_prompt=None):
            messages = super().prepare_inputs(inputs, system_prompt=system_prompt)
            self.prepared = messages
            return messages

        def generate_inner(self, *args, **kwargs):
            code, answer, response = super().generate_inner(*args, **kwargs)
            self.http_status = response.status_code
            return code, answer, response

    model = request["model"]
    key = os.environ["OPENAI_API_KEY"]
    model_options = {"retry": 1, "verbose": False, **options.get("model_options", {})}
    workers = request["concurrent"]
    # The native prompt builder may materialize media, so run it serially just as
    # the toolkit's image inference runner does.
    def prompts():
        for _, row in data.iterrows():
            index = str(row["index"])
            try:
                if dataset.MODALITY == "VIDEO":
                    prompt = dataset.build_prompt(row, video_llm=False)
                else:
                    prompt = dataset.build_prompt(row)
            except Exception as exc:
                print(f"Native prompt failed for sample {index}: {type(exc).__name__}", file=sys.stderr)
                yield index, None, "VLMEvalKitPromptError"
            else:
                yield index, prompt, None

    def infer(sample):
        index, prompt, error = sample
        started = datetime.now(timezone.utc).isoformat()
        vlm = RecordingVLM(model=model, key=key, api_base=api_url(request["base_url"]), **model_options)
        answer = INFER_FAIL_MSG
        if error is None:
            try:
                answer = vlm.generate(message=prompt, dataset=options["dataset"])
                if (not isinstance(answer, str) or not answer.strip() or vlm.fail_msg.strip() in answer
                        or (vlm.http_status is not None and not 200 <= vlm.http_status < 300)):
                    error = "VLMEvalKitInferenceError"
                    answer = INFER_FAIL_MSG
            except Exception as exc:
                error = type(exc).__name__
        trial = uuid4()
        target = job / str(trial)
        target.mkdir()
        result = {
            "id": str(trial), "task_name": index,
            "task_checksum": hashlib.sha256(
                json.dumps([options["dataset"], index, vlm.prepared], sort_keys=True).encode()).hexdigest(),
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "agent_info": {"name": "vlmevalkit", "model_info": {"name": model}},
            "exception_info": {"exception_type": error} if error else None,
            "vlmeval_result": {"dataset": options["dataset"], "sample_id": index,
                               "request_messages": vlm.prepared, "prediction": answer if error is None else None,
                               "http_status": vlm.http_status},
        }
        write_json(target / "result.json", result)
        return target / "result.json", answer, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(infer, prompts()))
    predictions = data.drop(columns=["image"], errors="ignore").copy()
    predictions["prediction"] = [answer for _, answer, _ in results]
    prediction_file = native / "predictions.xlsx"
    dump(predictions, str(prediction_file))
    success = all(error is None for _, _, error in results)
    if success and options["mode"] == "all":
        judge = dict(options.get("judge", {}))
        judge_key = os.getenv("VLMEVALKIT_JUDGE_API_KEY")
        judge_url = os.getenv("VLMEVALKIT_JUDGE_BASE_URL")
        if bool(judge_key) != bool(judge_url):
            raise ValueError("Supply both VLMEVALKIT_JUDGE_BASE_URL and VLMEVALKIT_JUDGE_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)
        if judge_key and judge_url:
            judge.update(key=judge_key, api_base=api_url(judge_url))
        score = dataset.evaluate(str(prediction_file), **judge)
        metrics = json.loads(score.to_json(orient="records")) if hasattr(score, "to_json") else score
        write_json(native / "metrics.json", metrics)
        details = scores_from_details(native, options)
        for receipt, _, error in results:
            row = json.loads(receipt.read_text())
            value = details.get(row["task_name"])
            if error is None and value is not None:
                row["verifier_result"] = {"rewards": {"reward": value}}
                write_json(receipt, row)
        if options.get("metric_key"):
            value = metric_result(metrics, options["metric_key"])
            write_json(job / "official-result.json", {
                "dataset": request["name"], "metric": options.get("metric", options["metric_key"]),
                "score": value, "denominator": len(data), "source": "vlmevalkit/metrics.json",
            })
    return int(not success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    raise SystemExit(worker(parser.parse_args().request))

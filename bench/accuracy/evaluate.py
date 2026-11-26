"""Run or rescore the fixed Hy4 Flash GSM8K-300 and RULER-16K cohorts."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from deploy.common import load, require_digest

DATA = Path(__file__).resolve().parent


def gsm_score(text, answer, lock):
    def normalize(value):
        for pattern in lock["normalize"]:
            value = re.sub(pattern, "", value)
        return value.lower()

    scores = {}
    for name, filter_spec in lock["filters"].items():
        matches = re.findall(filter_spec["regex_pattern"], text)
        match = matches[filter_spec.get("group_select", 0)] if matches else "[invalid]"
        if isinstance(match, tuple):
            match = next((entry for entry in match if entry), "[invalid]")
        prediction = normalize(match.strip())
        scores[name] = {"answer": prediction, "correct": prediction == normalize(answer)}
    return scores


def ruler_score(text, answers):
    clean = re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip().lower()
    hits = [answer.lower() in clean for answer in answers]
    return {"matched": sum(hits), "possible": len(hits)}


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_suite(suite, docs, responses, lock):
    indexed = {record["id"]: record for record in responses}
    if len(indexed) != len(responses) or set(indexed) != {doc["id"] for doc in docs}:
        raise ValueError("Responses must cover the exact cohort with one reply per case")
    totals = defaultdict(lambda: {"matched": 0, "possible": 0})
    for doc in docs:
        row = indexed[doc["id"]]
        usage = row["usage"]
        if (usage["prompt_tokens"] != len(doc["prompt"])
                or not 0 <= usage["completion_tokens"] <= doc["max_tokens"]
                or row["finish_reason"] not in ("stop", "length")):
            raise ValueError(f"Unexpected token counts or status for case {doc['id']}")
        if suite == "ruler":
            item = ruler_score(row["text"], doc["answers"])
            totals[doc["task"]]["matched"] += item["matched"]
            totals[doc["task"]]["possible"] += item["possible"]
        else:
            for name, item in gsm_score(row["text"], doc["answers"][0], lock["scorer"]).items():
                totals[name]["matched"] += item["correct"]
                totals[name]["possible"] += 1
    return {
        "suite": suite,
        "cases": len(docs),
        "scores": {task: {**counts, "accuracy": counts["matched"] / counts["possible"]}
                   for task, counts in totals.items()},
    }


def complete(base_url, doc, service, stop):
    payload = {
        "model": service["model"], "prompt": doc["prompt"], "max_tokens": doc["max_tokens"],
        "temperature": service["temperature"], "seed": service["seed"], "stop": stop,
        "return_token_ids": True,
    }
    request = Request(base_url.rstrip("/") + "/v1/completions",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=1800) as response:
        answer = json.load(response)
    choice, = answer["choices"]
    return {
        "id": doc["id"], "usage": answer["usage"], "text": choice["text"],
        "finish_reason": choice["finish_reason"], "token_ids": choice["token_ids"],
    }


def run(output, docs, suite, config, stop, base_url):
    service = config["service"]
    concurrency = config[suite]["concurrency"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as raw, ThreadPoolExecutor(max_workers=concurrency) as workers:
        for start in range(0, len(docs), concurrency):
            batch = docs[start:start + concurrency]
            tasks = [workers.submit(complete, base_url, doc, service, stop) for doc in batch]
            for task in tasks:
                raw.write(json.dumps(task.result(), ensure_ascii=False) + "\n")
                raw.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("gsm8k", "ruler"))
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--responses", type=Path, help="Rescore existing responses without an NPU")
    selection.add_argument("--output", type=Path, help="Generate fresh responses into this JSONL file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    lock = load(DATA / "inputs.json")[args.suite]
    require_digest(args.manifest, lock["manifest_sha256"])
    manifest = {doc["id"]: doc for doc in read_rows(args.manifest)}
    docs = [manifest[case] for case in lock["ids"]]
    if args.output is not None:
        config = load(DATA / "protocol.json")
        stop = lock["scorer"]["until"] if args.suite == "gsm8k" else []
        run(args.output, docs, args.suite, config, stop, args.base_url)
    result = score_suite(args.suite, docs, read_rows(args.output or args.responses), lock)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

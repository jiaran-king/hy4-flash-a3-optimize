"""Measure one fixed load through an existing HY4 BF16 service."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter
from urllib.request import urlopen

from deploy.common import ROOT, environment, model_identity, runtime_identity, settings
from .priming import prime
from .report import report

METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
}


def write_trace(path, workload, requests):
    width = math.ceil(workload["input_tokens"] / 128)
    with path.open("x") as stream:
        for index in range(requests):
            sample = {
                "timestamp": 0,
                "input_length": workload["input_tokens"],
                "output_length": workload["output_tokens"],
                "hash_ids": list(range(index * width, (index + 1) * width)),
            }
            stream.write(json.dumps(sample) + "\n")


def sample_server(base_url, output, stop):
    with output.open("x") as stream:
        while not stop.is_set():
            row = {"clock_s": perf_counter()}
            try:
                with urlopen(base_url + "/metrics", timeout=2) as response:
                    for line in response.read().decode().splitlines():
                        parts = line.split()
                        if len(parts) != 2:
                            continue
                        key = parts[0].split("{", 1)[0]
                        if key in METRICS:
                            row[key] = max(row.get(key, 0), float(parts[1]))
            except (OSError, ValueError) as error:
                row["sample_error"] = type(error).__name__
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(15)


def measure(output, model, base_url, workload, concurrency, requests, env):
    trace = output / "input-trace.jsonl"
    write_trace(trace, workload, requests)
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--endpoint", "/v1/completions", "--base-url", base_url,
        "--model", workload["served_model"], "--tokenizer", str(model),
        "--dataset-name", "timed_trace", "--dataset-path", str(trace),
        "--timed-trace-chunk-hash-size", "128", "--no-self-timed", "--request-rate", workload["request_rate"],
        "--num-prompts", str(requests), "--max-concurrency", str(concurrency),
        "--seed", str(workload["seed"]), "--temperature", str(workload["temperature"]), "--ignore-eos",
        "--num-warmups", "0", "--ready-check-timeout-sec", "0",
        "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,90,95",
        "--save-result", "--save-detailed", "--result-dir", str(output), "--result-filename", "raw.json",
    ]
    stop = Event()
    sampler = Thread(target=sample_server, args=(base_url, output / "service-metrics.jsonl", stop), daemon=True)
    sampler.start()
    try:
        with (output / "client.log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        stop.set()
        sampler.join()


def check_text(output, model, base_url, env):
    directory = ROOT / "bench/accuracy"
    path = output / "text-quality.json"
    subprocess.run([
        sys.executable, str(directory / "check_text.py"),
        "--baseline-tools", str(directory), "--base-url", base_url,
        "--tokenizer", str(model), "--output", str(path),
    ], env=env, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(path.read_text())
    if len(result["inherited"]) != 6 or not all(row["matched"] for row in result["inherited"]):
        raise ValueError("One or more of the six text checks failed")
    if not result["long"]["matched"] or result["long"]["usage"]["prompt_tokens"] != 16384:
        raise ValueError("The 16K retrieval check did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=9)
    parser.add_argument("--output-tokens", type=int, choices=(16, 128, 1024), default=1024)
    parser.add_argument("--history", choices=("baseline", "primed", "none"), default="baseline")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.requests:
        parser.error("Require 1 <= concurrency <= requests")
    model = args.model_dir.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    receipt = runtime_identity(runtime)
    checkpoint = model_identity(model)
    workload = settings("workload")
    env = environment(runtime, args.cann_root.resolve())
    env["PYTHONHASHSEED"] = str(workload["seed"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")
    if args.history == "primed":
        traces = output / "priming-trace.jsonl"
        prompts = output / "priming-prompts.json"
        write_trace(traces, workload, 4)
        subprocess.run([
            sys.executable, "-m", "bench.timed_prompts", str(traces), str(model),
            str(ROOT / "configs/workload.json"), "4", str(prompts), "--include-prompt-ids",
        ], cwd=ROOT, env=env, check=True)
        history = settings("priming")
        prepared = json.loads(prompts.read_text())
        if prepared["prompt_sha256"] != history["prompt_sha256"]:
            raise ValueError("The archived priming prompts did not reproduce")
        warmup = prime(base_url, workload, prepared["prompt_ids"], history["primed"])
        (output / "priming.json").write_text(json.dumps(warmup, indent=2) + "\n")
    if args.history != "none":
        check_text(output, model, base_url, env)
    workload["output_tokens"] = args.output_tokens
    measure(output, model, base_url, workload, args.concurrency, args.requests, env)
    if runtime_identity(runtime) != receipt:
        raise ValueError("The runtime receipt changed while measuring")
    score = report(output, workload)
    run = {
        "history": args.history,
        "runtime": receipt,
        "checkpoint": checkpoint,
        "workload": workload,
        "metrics": score,
    }
    (output / "result.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()

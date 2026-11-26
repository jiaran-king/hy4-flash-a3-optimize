"""Rebuild the actual pre-tokenized prompts from the pinned vLLM dataset."""

import argparse
import hashlib
import json
import os
from pathlib import Path

def expand(trace, model, plan, requests):
    from vllm.benchmarks.datasets.datasets import TimedTrace
    from vllm.tokenizers import get_tokenizer

    seed = plan["seed"]
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise ValueError("Start the dataset process with the frozen PYTHONHASHSEED")
    dataset = TimedTrace(
        dataset_path=str(trace), random_seed=seed, timed_trace_chunk_hash_size=128,
        timed_trace_label_timestamp="timestamp", timed_trace_label_input_length="input_length",
        timed_trace_label_output_length="output_length", timed_trace_label_hash_ids="hash_ids",
    )
    tokenizer = get_tokenizer(str(model))
    samples = dataset.sample(tokenizer, requests)
    if len(samples) != requests:
        raise ValueError("The pinned dataset produced the wrong number of prompts")
    for sample in samples:
        if (len(sample.prompt) != plan["input_tokens"]
                or sample.expected_output_len != plan["output_tokens"]):
            raise ValueError("The pinned dataset produced an unexpected shape")
    return samples


def fingerprint(samples):
    return [hashlib.sha256(json.dumps(sample.prompt, separators=(",", ":")).encode()).hexdigest()
            for sample in samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("requests", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-prompt-ids", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    samples = expand(args.trace, args.tokenizer, plan, args.requests)
    result = {"seed": plan["seed"], "prompt_sha256": fingerprint(samples)}
    if args.include_prompt_ids:
        result["prompt_ids"] = [sample.prompt for sample in samples]
    args.output.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

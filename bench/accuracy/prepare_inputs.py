"""Build immutable GSM8K and RULER input manifests for the pinned HY4 tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jinja2
from transformers import AutoTokenizer
import yaml


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def manifest(path, docs):
    with path.open("x") as stream:
        for doc in docs:
            stream.write(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")


def chat(tokenizer, body, reasoning_effort):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )


def gsm8k(folder, tokenizer, protocol):
    options = protocol["gsm8k"]
    config_path = folder / "upstream/lm-eval/gsm8k-cot.yaml"
    corpus = folder / "upstream/gsm8k/test.jsonl"
    config = yaml.safe_load(config_path.read_bytes())
    docs = rows(corpus)
    examples = config["fewshot_config"]["samples"]
    if config["task"] != options["task"] or len(examples) != options["few_shot"] or len(docs) != options["docs"]:
        raise ValueError("Unexpected upstream task or test split size")
    text_template = jinja2.Template(config["doc_to_text"])
    target_template = jinja2.Template(config["doc_to_target"])
    prefix = "\n\n".join(
        text_template.render(**doc) + " " + target_template.render(**doc) for doc in examples
    ) + "\n\n"
    order = sorted(range(len(docs)), key=lambda i: hashlib.sha256(
        f"{options['screen_seed']}:{i}".encode()).digest())
    screen = set(order[:options["screen_docs"]])
    reasoning = protocol["service"]["reasoning_effort"]
    stimuli = []
    for index, doc in enumerate(docs):
        body = prefix + text_template.render(**doc)
        token_ids = tokenizer.encode(chat(tokenizer, body, reasoning), add_special_tokens=False)
        stimuli.append({
            "id": index, "screen": index in screen, "prompt": token_ids,
            "answers": [target_template.render(**doc)], "max_tokens": options["max_tokens"],
        })
    if max(len(doc["prompt"]) + doc["max_tokens"] for doc in stimuli) > protocol["service"]["model_max_tokens"]:
        raise ValueError("At least one GSM8K problem exceeds the intended context")
    out = folder / "gsm8k.jsonl"
    manifest(out, stimuli)
    save(folder / "gsm8k-lock.json", {
        "sha256": {p.name: sha(p) for p in (config_path, corpus, out)},
        "selected_screen_ids": sorted(screen),
        "input_token_range": [min(len(d["prompt"]) for d in stimuli), max(len(d["prompt"]) for d in stimuli)],
        "until": config["generation_kwargs"]["until"],
        "filters": {f["name"]: f["filter"][0] for f in config["filter_list"]},
        "normalize": config["metric_list"][0]["regexes_to_ignore"],
    })
    print(f"GSM8K stored {len(stimuli)} docs and {len(screen)} screen IDs", flush=True)


def generate_ruler(source, output, model, protocol):
    conf = yaml.safe_load((source / "scripts/synthetic.yaml").read_bytes())
    data = source / "scripts/data"
    sys.path.insert(0, str(data / "synthetic"))
    from constants import TASKS

    for name, cap in protocol["subtasks"].items():
        task = conf[name]
        details = TASKS[task["task"]]
        if details["tokens_to_generate"] != cap:
            raise ValueError(f"Generation cap has changed for {name}")
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        if any(target.rglob("validation.jsonl")):
            continue
        command = [
            sys.executable, str(data / "synthetic" / f"{task['task']}.py"),
            "--save_dir", str(target), "--save_name", name, "--subset", "validation",
            "--tokenizer_path", str(model), "--tokenizer_type", "hf",
            "--max_seq_length", str(protocol["source_max_tokens"]),
            "--tokens_to_generate", str(cap), "--num_samples", str(protocol["docs_per_task"]),
            "--random_seed", str(protocol["seed"]),
            "--template", details["template"] + details["answer_prefix"],
        ]
        for key, value in task["args"].items():
            command.extend((f"--{key}", str(value)))
        # Mirror the upstream prepare.py dispatch without its unconditional NLTK download.
        subprocess.run(command, check=True)


def ruler(folder, model, tokenizer, protocol):
    import os

    ruler_config = protocol["ruler"]
    upstream = folder / "upstream"
    source, = (upstream / "ruler").glob("RULER-*")
    raw = folder / "ruler-raw"
    essays = upstream / "PaulGrahamEssays.json"
    pathname = source / "scripts/data/synthetic/json/PaulGrahamEssays.json"
    if not pathname.exists():
        os.symlink(essays, pathname)
    generate_ruler(source, raw, model, ruler_config)

    reasoning = protocol["service"]["reasoning_effort"]
    target_length = ruler_config["input_tokens"]
    pad, = tokenizer.encode("\n", add_special_tokens=False)
    marker = "RULER_HY4_PLACEHOLDER"
    opening = chat(tokenizer, marker, reasoning).split(marker, maxsplit=1)[0]
    offset = len(tokenizer.encode(opening, add_special_tokens=False))
    stimuli = []
    for name, cap in ruler_config["subtasks"].items():
        document, = (raw / name).rglob("validation.jsonl")
        docs = rows(document)
        if len(docs) != ruler_config["docs_per_task"]:
            raise ValueError(f"RULER generated {len(docs)} docs for {name}")
        for index, doc in enumerate(docs):
            rendered = chat(tokenizer, doc["input"], reasoning) + doc["answer_prefix"]
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            padding = target_length - len(ids)
            if not 0 <= padding <= 1024:
                raise ValueError(f"Unexpected padding {padding} for {name}/{index}")
            if tokenizer.decode(ids[:offset], skip_special_tokens=False).rstrip() != opening.rstrip():
                raise ValueError("The HY4 user-content boundary differs from the template")
            ids[offset:offset] = [pad] * padding
            stimuli.append({
                "id": f"{name}/{index:03}", "task": name, "answers": doc["outputs"],
                "prompt": ids, "max_tokens": cap, "padding_tokens": padding,
            })
    out = folder / "ruler.jsonl"
    manifest(out, stimuli)
    save(folder / "ruler-lock.json", {
        "prompt_sha256": sha(out), "input_tokens": target_length, "essay_sha256": sha(essays),
        "padding": [min(d["padding_tokens"] for d in stimuli), max(d["padding_tokens"] for d in stimuli)],
    })
    print(f"RULER stored {len(stimuli)} docs at {target_length} tokens each", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("gsm8k", "ruler"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    settings = read(args.folder / "protocol.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.suite == "gsm8k":
        gsm8k(args.folder, tokenizer, settings)
    else:
        ruler(args.folder, args.model, tokenizer, settings)

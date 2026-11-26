"""Run the inherited six cases and check retrieval across a 16K prompt."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


FILLER = (
    "The following notes describe shipments, inventories, and labels. "
    "These are unrelated notes. "
)
TARGET_TOKENS = 16384
MINIMUM_TOKENS = 16320
EXPECTED_PROJECT = 6847


def long_prompt(tokenizer):
    prefix = f"The project number is {EXPECTED_PROJECT}. Remember that number. "
    suffix = "What is the project number? Respond with only the integer."
    low, high = 0, TARGET_TOKENS
    while low < high:
        count = (low + high + 1) // 2
        prompt = prefix + FILLER * count + suffix
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            reasoning_effort="no_think",
        )
        size = len(tokenizer.encode(rendered, add_special_tokens=False))
        if size <= TARGET_TOKENS:
            low = count
        else:
            high = count - 1
    return prefix + FILLER * low + suffix


def check_long(base_url, tokenizer_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = long_prompt(tokenizer)
    payload = {
        "model": "hy4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    output = (choice["message"].get("content") or "").strip()
    numbers = re.findall(r"-?\d[\d,]*", output)
    actual = int(numbers[-1].replace(",", "")) if numbers else None
    prompt_tokens = result["usage"]["prompt_tokens"]
    return {
        "case": "long-retrieval",
        "expected": EXPECTED_PROJECT,
        "actual": actual,
        "matched": actual == EXPECTED_PROJECT and MINIMUM_TOKENS <= prompt_tokens <= TARGET_TOKENS,
        "output": output,
        "finish_reason": choice["finish_reason"],
        "usage": result["usage"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tools", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.baseline_tools))
    from check_service import check

    references = list(check(args.base_url))
    extended = check_long(args.base_url, args.tokenizer)
    data = {"inherited": references, "long": extended}
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": sum(case["matched"] for case in references), "long": extended}))


if __name__ == "__main__":
    main()

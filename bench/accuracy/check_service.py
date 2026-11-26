"""Fixed text-generation checks for the HY4 Flash baseline."""

import argparse
import json
import re
import urllib.request


CASES = (
    ("Respond with only the integer. What is 37 + 58?", 95),
    ("Respond with only the integer. What is 135 times 17?", 2295),
    ("Respond with only the integer. What is the remainder when 319 is divided by 12?", 7),
    ("只输出整数。一个盒子里有 28 支笔，拿走 7 支，还剩多少支？", 21),
    ("Respond with only the integer. A warehouse starts with 48 boxes and receives 76 more. How many boxes are in the warehouse?", 124),
    (
        "The project number is 6847. Remember that number. "
        + "The following notes describe shipments, inventories, and labels. These are unrelated notes. " * 150
        + "What is the project number? Respond with only the integer.",
        6847,
    ),
)


def check(base_url: str):
    for index, (prompt, expected) in enumerate(CASES, 1):
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
        yield {
            "case": index, "expected": expected, "actual": actual,
            "matched": actual == expected, "output": output,
            "finish_reason": choice.get("finish_reason"), "usage": result.get("usage"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    results = []
    for result in check(args.base_url):
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(record["matched"] for record in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

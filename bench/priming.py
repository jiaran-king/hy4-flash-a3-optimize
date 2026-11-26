"""Run the short and mixed request history recorded for the final candidate."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from threading import Barrier, Event
from time import perf_counter
from urllib.request import Request, urlopen


def stream(base_url, workload, prompt, output_tokens, reached=None, barrier=None):
    body = {
        "model": workload["served_model"], "prompt": prompt, "max_tokens": output_tokens,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
        "temperature": workload["temperature"],
    }
    request = Request(base_url + "/v1/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    start = perf_counter()
    times = []
    usage = finish = None
    with urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            message = json.loads(line[6:])
            if message.get("error"):
                raise RuntimeError("The priming request returned a stream error")
            usage = message.get("usage") or usage
            if message.get("choices"):
                finish = message["choices"][0].get("finish_reason") or finish
                times.append(perf_counter())
                if reached is not None and len(times) == 16:
                    reached.set()
    if (usage is None or usage["prompt_tokens"] != workload["input_tokens"]
            or usage["completion_tokens"] != output_tokens
            or finish != "length" or len(times) != output_tokens):
        raise ValueError("The priming request did not produce the requested token stream")
    return {"start": start, "first": times[0], "last": times[-1], "output_tokens": output_tokens}


def prime(base_url, workload, prompts, history):
    summaries = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for stage in history:
            count, limit = stage["requests"], stage["output_tokens"]
            barrier = Barrier(count)
            tasks = [pool.submit(stream, base_url, workload, prompt, limit, None, barrier)
                     for prompt in prompts[:count]]
            summaries.append([task.result() for task in tasks])
        opened = [Event() for _ in prompts[:3]]
        holders = [pool.submit(stream, base_url, workload, prompt, 128, ready)
                   for prompt, ready in zip(prompts[:3], opened, strict=True)]
        while not all(event.is_set() for event in opened):
            done, _ = wait(holders, timeout=0.1, return_when=FIRST_COMPLETED)
            if done:
                for task in done:
                    task.result()
                raise ValueError("A priming decoder finished before the mixed request was ready")
        inserted = stream(base_url, workload, prompts[3], 16)
        mixed = [task.result() for task in holders]
    if not all(row["first"] < inserted["start"] < inserted["first"] < row["last"] for row in mixed):
        raise ValueError("Mixed priming did not overlap the three decode streams")
    return {"stages": summaries, "mixed": [*mixed, inserted]}

"""Audit timed-trace measurements using the Case 23 steady-stage window."""

import argparse
from bisect import bisect_left
from itertools import accumulate
import json
import math
from pathlib import Path
from statistics import fmean


def percentile(values, fraction):
    numbers = sorted(values)
    position = (len(numbers) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)


def steady_interval(rows, concurrency):
    # Reproduce the stable-stage selection from AISBench 8b80c4e. The first
    # time the client fills is the initial ramp. The next refill opens the window.
    events = []
    for row in rows:
        events.extend(((row["start_s"], 1), (row["end_s"], -1)))
    events.sort(key=lambda event: event[0])
    active = 0
    refill = 0
    start = end = None
    peak = 0
    for stamp, delta in events:
        active += delta
        peak = max(peak, active)
        if active == concurrency:
            if delta == 1:
                refill += 1
                if refill == 2:
                    start = stamp
            end = stamp
    if start is None or end <= start:
        return None, peak
    return (start, end), peak


def distributions(rows):
    metrics = {}
    for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
        values = [row[name] for row in rows]
        metrics[name] = {
            "mean": fmean(values), "p90": percentile(values, 0.90), "p95": percentile(values, 0.95),
        }
    return metrics


def peak_service_usage(path):
    if not path.is_file():
        return {}
    peaks = {}
    for line in path.read_text().splitlines():
        for name, value in json.loads(line).items():
            if name.startswith("vllm:"):
                peaks[name] = max(peaks.get(name, 0), value)
    return peaks


def result_rows(raw, plan):
    requested = raw["num_prompts"]
    expected_input, expected_output = plan["input_tokens"], plan["output_tokens"]
    if (raw["completed"] != requested or raw["failed"] != 0
            or raw["input_lens"] != [expected_input] * requested
            or raw["output_lens"] != [expected_output] * requested):
        raise ValueError("Incomplete requests or mismatched token lengths")
    origin = min(raw["start_times"])
    rows = []
    fields = zip(raw["start_times"], raw["latencies"], raw["ttfts"], raw["itls"], strict=True)
    for index, (timestamp, latency, ttft, gaps) in enumerate(fields):
        if len(gaps) + 1 != expected_output:
            raise ValueError(f"Request {index} has missing per-token timestamps")
        start = timestamp - origin
        first_token = start + ttft
        times = [first_token, *(first_token + elapsed for elapsed in accumulate(gaps))]
        if not all(math.isfinite(value) for value in times) or not 0 < ttft < latency:
            raise ValueError(f"Request {index} has invalid token timings")
        if any(gap < 0 for gap in gaps) or times[-1] > start + latency + 1e-5:
            raise ValueError(f"Request {index} has inconsistent stream timing")
        rows.append({
            "request": index, "start_s": start, "end_s": start + latency,
            "ttft_ms": ttft * 1000, "tpot_ms": (latency - ttft) * 1000 / (expected_output - 1),
            "e2el_ms": latency * 1000, "input_tokens": expected_input, "output_tokens": expected_output,
            "output_timestamps_s": times,
        })
    if len(rows) != requested:
        raise ValueError("Request timing count differs from the workload")
    return rows


def summarize(directory, plan):
    raw = json.loads((directory / "raw.json").read_text())
    rows = result_rows(raw, plan)
    concurrency = raw["max_concurrency"]
    window, peak = steady_interval(rows, concurrency)
    total_duration = max(row["end_s"] for row in rows) - min(row["start_s"] for row in rows)
    complete = {
        "duration_s": total_duration, "requests": len(rows),
        "output_tok_s": sum(row["output_tokens"] for row in rows) / total_duration,
        **distributions(rows),
    }
    metrics = {
        "concurrency": concurrency, "peak_client_concurrency": peak, "completed": len(rows),
        "full": complete, "steady": None,
        "server_peaks": peak_service_usage(directory / "service-metrics.jsonl"),
    }
    if window:
        start, end = window
        duration = end - start
        cohort = [row for row in rows if start <= row["start_s"] <= end]
        for row in cohort:
            row["aisbench_steady_selected"] = True
        emitted = 0
        for row in rows:
            times = row["output_timestamps_s"]
            emitted += bisect_left(times, end) - bisect_left(times, start)
        metrics["steady"] = {
            "start_s": start, "end_s": end, "duration_s": duration,
            "requests": len(cohort), "emitted_tokens": emitted, "output_tok_s": emitted / duration,
            "aisbench_start_selected_tok_s": sum(row["output_tokens"] for row in cohort) / duration,
            **distributions(cohort),
        }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    metrics, records = summarize(args.directory, json.loads(args.plan.read_text()))
    with (args.directory / "requests.jsonl").open("x") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    with (args.directory / "metrics.json").open("x") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

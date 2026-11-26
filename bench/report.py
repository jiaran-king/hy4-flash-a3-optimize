"""Recompute HY4 Flash service measurements from a local benchmark directory."""

import argparse
import json
from pathlib import Path

from deploy.common import settings
from .reduce_measurement import percentile, summarize


def report(directory, workload):
    score, rows = summarize(directory, workload)
    wave = sorted(rows, key=lambda row: row["start_s"])[:score["concurrency"]]
    score["first_wave_max_ttft_ms"] = max(row["ttft_ms"] for row in wave)
    raw = json.loads((directory / "raw.json").read_text())
    gaps = [1000 * gap for stream in raw["itls"] for gap in stream]
    score["itl_ms"] = {"p99": percentile(gaps, 0.99), "max": max(gaps)}
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+")
    parser.add_argument("--output-tokens", type=int, choices=(16, 128, 1024))
    args = parser.parse_args()
    workload = settings("workload")
    if args.output_tokens is not None:
        workload["output_tokens"] = args.output_tokens
    print(json.dumps({str(path): report(path, workload) for path in args.run}, indent=2))


if __name__ == "__main__":
    main()

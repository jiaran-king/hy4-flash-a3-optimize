"""Start one A3 HY4 Flash BF16 service with the common frozen workload settings."""

import argparse
import json
import os
from pathlib import Path
import sys

from .common import environment, model_identity, runtime_identity, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workload = settings("workload")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
        str(args.model_dir.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", workload["served_model"],
        "--tensor-parallel-size", str(workload["tensor_parallel_size"]), "--enable-expert-parallel",
        "--dtype", workload["dtype"], "--kv-cache-dtype", workload["kv_cache_dtype"],
        "--kv-cache-memory-bytes", str(workload["kv_cache_bytes_per_die"]),
        "--max-model-len", str(workload["model_max_tokens"]),
        "--max-num-batched-tokens", str(workload["max_batch_tokens"]),
        "--max-num-seqs", str(workload["max_sequences"]), "--load-format", "safetensors",
        "--enforce-eager", "--no-enable-prefix-caching", "--engram-config", '{"cpu_offload":true}',
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
    ]
    receipt = runtime_identity(args.runtime.resolve(), require_device=not args.dry_run)
    if args.dry_run:
        command[4] = "<MODEL_PATH>"
        print(json.dumps({"stage": receipt["stage"], "command": command}, indent=2))
        return
    model_identity(args.model_dir.resolve(strict=True))
    os.execve(sys.executable, command, environment(args.runtime.resolve(), args.cann_root.resolve()))


if __name__ == "__main__":
    main()

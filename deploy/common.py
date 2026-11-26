"""Shared configuration and runtime identity checks."""

import hashlib
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("workspace/runtime-01")


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_digest(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual}")
    return actual


def settings(name):
    return load(ROOT / "configs" / (name + ".json"))


def model_identity(model):
    lock = settings("runtime")["model"]
    config = require_digest(model / "config.json", lock["config_sha256"])
    index_file = model / "model.safetensors.index.json"
    index = require_digest(index_file, lock["index_sha256"])
    template = require_digest(model / "chat_template.jinja", lock["chat_template_sha256"])
    tokenizer = require_digest(model / "tokenizer.json", lock["tokenizer_sha256"])
    shards = set(load(index_file)["weight_map"].values())
    if len(shards) != lock["safetensors_shards"] or any(
        Path(shard).name != shard or not (model / shard).is_file() for shard in shards
    ):
        raise ValueError("Checkpoint shard inventory differs from the pinned index")
    return {
        "config_sha256": config, "index_sha256": index, "shards": len(shards),
        "chat_template_sha256": template, "tokenizer_sha256": tokenizer,
    }


def runtime_identity(root, *, require_device=True):
    receipt = load(root / "runtime.json")
    runtime = root / RUNTIME
    for path, expected in receipt["source_files"].items():
        require_digest(runtime / path, expected)
    if require_device:
        lock = settings("runtime")["native_extension"]
        if receipt["native_extension_sha256"] != lock["sha256"]:
            raise ValueError("This runtime was prepared in source-only mode")
        require_digest(runtime / "vllm-ascend/vllm_ascend" / lock["file"], lock["sha256"])
    return receipt


def environment(root, cann_root):
    runtime = root / RUNTIME
    acl_path = cann_root / "python/site-packages"
    if not (acl_path / "acl/__init__.py").is_file():
        raise ValueError("Pass the CANN toolkit directory containing the Python ACL package")
    python_paths = [runtime / "vllm", runtime / "vllm-ascend", acl_path]
    previous = os.environ.get("PYTHONPATH")
    if previous:
        python_paths.append(previous)
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, python_paths)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_MLA_DISABLE": "1",
        "VLLM_USE_RUST_FRONTEND": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }

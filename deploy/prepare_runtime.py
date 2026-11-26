"""Prepare the pinned A3 Python overlay and optionally attach native A3 libraries."""

import argparse
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .common import ROOT, RUNTIME, require_digest, settings, sha256


def archives_into(runtime, downloads, cache, sources):
    for name, lock in sources.items():
        archive = (cache or downloads) / lock["archive"]
        if cache is None:
            with urlopen(lock["archive_url"], timeout=120) as response, archive.open("xb") as out:
                shutil.copyfileobj(response, out)
        require_digest(archive, lock["sha256"])
        extracted = downloads / name
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(extracted, filter="data")
        top, = extracted.iterdir()
        top.rename(runtime / name)


def git_apply(root, path):
    git_environment = {**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent)}
    command = ["git", "apply", "--whitespace=nowarn"]
    subprocess.run([*command, "--check", str(path)], cwd=root, env=git_environment, check=True)
    subprocess.run([*command, str(path)], cwd=root, env=git_environment, check=True)


def port(root, lock):
    baseline = ROOT / "patches/a3-baseline"
    runtime = root / RUNTIME
    integration = baseline / "ascend-integration.patch"
    require_digest(integration, lock["ascend_integration"]["sha256"])
    git_apply(runtime / "vllm-ascend", integration)
    for script, helper in (
        ("apply_a3_port.py", "a3_sparse_attention.py"),
        ("apply_a3_indexpool.py", "a3_indexpool_fp8.py"),
    ):
        subprocess.run([sys.executable, str(baseline / script), str(runtime), str(baseline / helper)],
                       check=True)
    for path, expected in lock["baseline_files"].items():
        require_digest(runtime / path, expected)


def selected_series(profile, through):
    series = settings(profile)["series"]
    if series is None:
        if through:
            raise ValueError("--through requires the optimized profile")
        return ()
    series = ROOT / series
    names = [Path(line) for line in series.read_text().splitlines() if line and not line.startswith("#")]
    stages = tuple(dict.fromkeys(name.parent.name for name in names))
    selected = set(stages[:stages.index(through) + 1] if through else stages)
    return tuple(series.parent / name for name in names if name.parent.name in selected)


def changed_sources(patch):
    prefix = (Path("b") / RUNTIME).as_posix() + "/"
    return [line[len("+++ ") + len(prefix):]
            for line in patch.read_text().splitlines() if line.startswith("+++ " + prefix)]


def assert_postimages(root, patch):
    # Git-format diffs carry abbreviated object IDs of the original archived postimages.
    # Plain unified diffs are checked for applicability with git apply.
    expected = None
    for line in patch.read_text().splitlines():
        if line.startswith("diff --git "):
            expected = None
        elif line.startswith("index "):
            match = re.match(r"index [a-f0-9]+\.\.([a-f0-9]+)", line)
            expected = match[1] if match else None
        elif line.startswith("+++ b/") and expected:
            path = root / line.removeprefix("+++ b/")
            actual = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
            if not actual.startswith(expected):
                raise ValueError(f"Archived patch postimage mismatch for {path.name}")
            expected = None


def attach_native(runtime, native, lock):
    source = native / lock["file"]
    digest = require_digest(source, lock["sha256"])
    target = runtime / "vllm-ascend/vllm_ascend"
    for relative in (lock["file"], "_cann_ops_custom/vendors", "lib"):
        artifact = (native / relative).resolve(strict=True)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(artifact, target_is_directory=artifact.is_dir())
    return digest


def prepare(output, profile, cache, native, through):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    lock = settings("runtime")
    patches = selected_series(profile, through)
    missing = [patch.parent.name for patch in patches if not patch.is_file()]
    if missing:
        raise ValueError(f"Check out optimized or its stage branch to use {missing[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".hy4-prepare-") as scratch:
        clean = Path(scratch) / "overlay"
        runtime = clean / RUNTIME
        runtime.mkdir(parents=True)
        archives_into(runtime, Path(scratch), cache, lock["sources"])
        port(clean, lock)
        tracked = set(lock["baseline_files"])
        for patch in patches:
            git_apply(clean, patch)
            assert_postimages(clean, patch)
            tracked.update(changed_sources(patch))
        digest = attach_native(runtime, native, lock["native_extension"]) if native else None
        receipt = {
            "profile": profile,
            "stage": patches[-1].parent.name if patches else "run-03",
            "source_files": {path: sha256(runtime / path) for path in sorted(tracked)},
            "native_extension_sha256": digest,
        }
        (clean / "runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
        clean.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archives", type=Path, help="Optional directory with the locked source archives")
    parser.add_argument("--through", help="Last stage to apply, e.g. 02-query-chunk-128")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--native-root", type=Path, help="vllm_ascend package root from the reference A3 image")
    mode.add_argument("--source-only", action="store_true", help="Check source applicability on a CPU host")
    args = parser.parse_args()
    result = prepare(args.output.resolve(), args.profile, args.archives, args.native_root, args.through)
    print(f"Prepared {result['stage']} at {args.output} ({len(result['source_files'])} checked source files)")


if __name__ == "__main__":
    main()

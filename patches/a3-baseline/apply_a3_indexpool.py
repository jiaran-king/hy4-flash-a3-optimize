"""Keep HY4's FP8 IndexPool scoring semantics with A3-compatible tensors."""

import sys
from pathlib import Path


def apply(runtime: Path, rounding_source: Path) -> None:
    upstream = runtime / "vllm/vllm/models/hy_v4_flash/nvidia/indexpool_ops.py"
    models = runtime / "vllm-ascend/vllm_ascend/models"
    asc = models / "hy_v4_flash_long.py"
    helper = models / "hy_v4_flash_a3_fp8.py"
    original = upstream.read_text()
    port = asc.read_text()
    old_signature = "def fwht_quantize(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:"
    split_signature = old_signature.replace("fwht_quantize", "fwht_scaled")
    old_cast = (
        "    quant = (x / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)\n"
        "    return quant, scale"
    )
    cast_wrapper = (
        "    return (x / scale[:, None]).clamp(-FP8_MAX, FP8_MAX), scale\n\n\n"
        + old_signature + "\n"
        "    values, scale = fwht_scaled(vectors)\n"
        "    return values.to(torch.float8_e4m3fn), scale"
    )
    if split_signature not in original:
        assert original.count(old_signature) == original.count(old_cast) == 1
        original = original.replace(old_signature, split_signature).replace(old_cast, cast_wrapper)

    old_roundtrip = (
        "    quantized, scales = fwht_quantize(vectors.reshape(-1, INDEX_DIM).bfloat16())\n"
        "    return (quantized.float() * scales[:, None]).to(torch.bfloat16).view_as(vectors)"
    )
    new_roundtrip = (
        "    values, scales = fwht_scaled(vectors.reshape(-1, INDEX_DIM).bfloat16())\n"
        "    quantized = round_e4m3fn(values)\n"
        "    return (quantized * scales[:, None]).to(torch.bfloat16).view_as(vectors)"
    )
    anchor = "from vllm_ascend.models.hy_v4_flash_a3 import attend_paged_with_sink"
    if new_roundtrip not in port:
        assert port.count(old_roundtrip) == port.count(anchor) == 1
        port = port.replace("    fwht_quantize,\n", "    fwht_scaled,\n")
        port = port.replace(anchor, anchor + "\nfrom vllm_ascend.models.hy_v4_flash_a3_fp8 import round_e4m3fn")
        port = port.replace(old_roundtrip, new_roundtrip)

    compile(original, str(upstream), "exec")
    compile(port, str(asc), "exec")
    helper.write_bytes(rounding_source.read_bytes())
    upstream.write_text(original)
    asc.write_text(port)


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

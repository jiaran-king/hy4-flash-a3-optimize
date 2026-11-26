"""Install the isolated A3 attention path over the verified A5 integration."""

import sys
from pathlib import Path


def apply(runtime: Path, sparse_path: Path) -> None:
    models = runtime / "vllm-ascend/vllm_ascend/models"
    target = models / "hy_v4_flash_long.py"
    original = target.read_text()
    marker = "from vllm_ascend.models.hy_v4_flash_a3 import attend_paged_with_sink"
    if marker in original:
        if (models / "hy_v4_flash_a3.py").read_bytes() != sparse_path.read_bytes():
            raise ValueError("The installed A3 sparse reference differs from this patch")
        return

    # Match the FP32 IndexPool scoring correction from the subsequent upstream PR.
    old_score = "projected = q.reshape(-1, INDEX_DIM) @ keys.T"
    assert original.count(old_score) == 1
    updated = original.replace(old_score, "projected = q.reshape(-1, INDEX_DIM).float() @ keys.float().T")
    import_anchor = "from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window"
    assert updated.count(import_anchor) == 1
    updated = updated.replace(import_anchor, import_anchor + "\n" + marker)

    first = updated.index("        attended, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(")
    last = "        output[:tokens].copy_((value.float() * sink_weight[..., None]).flatten(1))"
    end = updated.index(last, first) + len(last)
    replacement = """        attended = attend_paged_with_sink(
            q_absorbed,
            q_rope,
            kv_cache,
            rope_cache,
            topk,
            metadata.block_tables,
            metadata.actual_seq_lengths_q,
            metadata.seq_lens_list,
            attention.learnable_sink_param,
            attention.scaling,
        )
        value = torch.bmm(attended.transpose(0, 1), attention._value_projection).transpose(0, 1)
        output[:tokens].copy_(value.flatten(1))"""
    updated = updated[:first] + replacement + updated[end:]
    compile(updated, str(target), "exec")
    (models / "hy_v4_flash_a3.py").write_bytes(sparse_path.read_bytes())
    target.write_text(updated)


if __name__ == "__main__":
    apply(Path(sys.argv[1]), Path(sys.argv[2]))

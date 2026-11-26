# 批量 Prefill 选组与四头滑窗

一起处理 Prefill 选组并启用已核对的四头滑窗稀疏路径。四头与目标 TP16 形状对应。

## 复现

基于 `opt/05-batched-indexpool-decode` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`batch-prefill-indexpool-selection.patch`](batch-prefill-indexpool-selection.patch)
2. [`fuse-swa-prefill-sparse-attention.patch`](fuse-swa-prefill-sparse-attention.patch)
3. [`enable-four-head-swa-sparse.patch`](enable-four-head-swa-sparse.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 06-prefill-selection-swa` 重建到该阶段。

## 验证

完整服务按组合计分，分别检查选组与滑窗数值。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

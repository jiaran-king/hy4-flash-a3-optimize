# 扩大 Prefill query 分块

每次提交的 query token 数由 16 提到 128，减少 Prefill 提交次数。使用上一阶段的归档对照，核对首 token 和 A3 峰值分配。

## 复现

基于 `opt/01-batch-paged-decode` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`query-chunk-128-on-batch-decode.patch`](query-chunk-128-on-batch-decode.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 02-query-chunk-128` 重建到该阶段。

## 验证

完整服务、Prefill 算子耗时和显存对照。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

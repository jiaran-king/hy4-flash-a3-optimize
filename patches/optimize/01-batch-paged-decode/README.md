# 批量分页 Decode

按请求行读取各自的分页 KV，再批量提交注意力。保留每行有效位置、滑窗和 attention sink 语义。

## 复现

基于 `main` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`batch-paged-decode.patch`](batch-paged-decode.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 01-batch-paged-decode` 重建到该阶段。

## 验证

完整服务与分页算子对照。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

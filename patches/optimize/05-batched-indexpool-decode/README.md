# 批量 Decode IndexPool

合并并发 Decode 请求的 IndexPool 工作，保留每条请求的历史组选择和尾组位置。

## 复现

基于 `opt/04-native-sparse-prefill` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`batch-indexpool-decode.patch`](batch-indexpool-decode.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 05-batched-indexpool-decode` 重建到该阶段。

## 验证

完整服务与选组一致性对照。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

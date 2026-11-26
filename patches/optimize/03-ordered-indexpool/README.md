# IndexPool 有序选组与因果尾组

在设备端排序可精确表示的组 ID，并直接写入有效前缀和因果尾组。

## 复现

基于 `opt/02-query-chunk-128` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`indexpool-order-and-tail.patch`](indexpool-order-and-tail.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 03-ordered-indexpool` 重建到该阶段。

## 验证

完整服务、选组顺序和有效位逐项核对。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

# 稳定 ACLNN Host 缓存

在适用的 HY4 A3 eager 启动路径设置 Host 算子缓存默认值，减少启动请求历史引起的算子重新准备。

## 复现

基于 `opt/06-prefill-selection-swa` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`stabilize-hy4-aclnn-host-cache.patch`](stabilize-hy4-aclnn-host-cache.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 07-aclnn-host-cache` 重建到该阶段。

## 验证

两种启动历史短测与完整服务。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

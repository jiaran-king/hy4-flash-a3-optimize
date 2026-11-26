# 小批 Decode 本地专家与 FP32 归并

使用 TP 中已经复制的输入计算本地专家，再对 FP32 路由输出做一次归并。非适用批次沿用原路径。

## 复现

基于 `opt/08-fused-small-decode` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`replicate-hy4-small-decode-moe-reduced.patch`](replicate-hy4-small-decode-moe-reduced.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 09-local-expert-decode` 重建到该阶段。

## 验证

完整服务、路由归属和加权求和验证。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

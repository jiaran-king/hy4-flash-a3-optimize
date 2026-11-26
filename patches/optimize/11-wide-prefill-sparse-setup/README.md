# 融合宽批 iHC 并复用稀疏 Prefill 准备

融合宽批 iHC 与 IndexPool，将量化核的行数设为运行时参数；在同一次 forward 内复用滑窗页位、因果索引和长度张量。

## 复现

基于 `opt/10-joint-expert-reduce` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`fuse-hy4-wide-prefill-ihc.patch`](fuse-hy4-wide-prefill-ihc.patch)
2. [`fuse-hy4-prefill-indexpool-native-scale.patch`](fuse-hy4-prefill-indexpool-native-scale.patch)
3. [`reuse-hy4-prefill-quant-kernels.patch`](reuse-hy4-prefill-quant-kernels.patch)
4. [`reuse-hy4-sparse-prefill-setup.patch`](reuse-hy4-sparse-prefill-setup.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 11-wide-prefill-sparse-setup` 重建到该阶段。

## 验证

两种启动历史短测及两轮完整服务；另保留整模型生成差异。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

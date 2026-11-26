# 融合 IndexPool 和 iHC，配对归并专家输出

融合小批 IndexPool 与 iHC 的子算子，用模型 TP 组配对归并路由专家和共享专家的输出。

## 复现

基于 `opt/09-local-expert-decode` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`fuse-hy4-indexpool-rotary-quant.patch`](fuse-hy4-indexpool-rotary-quant.patch)
2. [`fuse-hy4-small-indexpool-stages.patch`](fuse-hy4-small-indexpool-stages.patch)
3. [`fuse-hy4-ihc-decode-projection.patch`](fuse-hy4-ihc-decode-projection.patch)
4. [`fuse-hy4-ihc-residual-handoff.patch`](fuse-hy4-ihc-residual-handoff.patch)
5. [`pack-hy4-decode-expert-tp-group-reductions.patch`](pack-hy4-decode-expert-tp-group-reductions.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 10-joint-expert-reduce` 重建到该阶段。

## 验证

完整服务按组合计分；记录浮点归并及 BF16 回写变化。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

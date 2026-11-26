# 原生稀疏 Prefill 与混合步 Decode

整理带掩码的 KV 位置、复用滑窗跨度、保留量化舍入，接入 BF16 原生稀疏算子。混合步同时处理 Decode 请求。

## 复现

基于 `opt/03-ordered-indexpool` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`spread-masked-kv-slots.patch`](spread-masked-kv-slots.patch)
2. [`share-swa-kv-tiles.patch`](share-swa-kv-tiles.patch)
3. [`native-bf16-prefill-bmm.patch`](native-bf16-prefill-bmm.patch)
4. [`keep-indexpool-rounding-on-vector-cores.patch`](keep-indexpool-rounding-on-vector-cores.patch)
5. [`share-swa-span.patch`](share-swa-span.patch)
6. [`fuse-long-prefill-sparse-attention.patch`](fuse-long-prefill-sparse-attention.patch)
7. [`batch-decode-prefix-in-mixed-steps.patch`](batch-decode-prefix-in-mixed-steps.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 04-native-sparse-prefill` 重建到该阶段。

## 验证

完整服务按这一组实现计分，保留算子与短测边界。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

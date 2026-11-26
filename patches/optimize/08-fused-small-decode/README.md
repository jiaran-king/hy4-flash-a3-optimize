# 融合小批 Decode 算子并复用索引

合并 iHC、IndexPool 量化和分页注意力的设备工作，复用同一步的页表与索引元数据。

## 复现

基于 `opt/07-aclnn-host-cache` 的已应用补丁。按以下顺序应用本目录补丁。

1. [`fuse-hy4-small-batch-ihc.patch`](fuse-hy4-small-batch-ihc.patch)
2. [`fuse-hy4-indexpool-quant.patch`](fuse-hy4-indexpool-quant.patch)
3. [`fuse-hy4-paged-decode.patch`](fuse-hy4-paged-decode.patch)
4. [`reuse-hy4-step-index-metadata.patch`](reuse-hy4-step-index-metadata.patch)

用 `deploy/prepare_runtime.py --profile optimized --through 08-fused-small-decode` 重建到该阶段。

## 验证

完整服务按组合计分，子项有算子或短测证据。压测条件和采集口径见 [测试说明](../../../docs/benchmark.md)。

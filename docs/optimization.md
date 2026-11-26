# 优化顺序

每阶段的补丁和应用说明在对应分支。每个比较链接显示本阶段新增的文件。
完整补丁链在 [optimized 分支](https://github.com/jiaran-king/hy4-flash-a3-optimize/tree/optimized)。

| 阶段 | 优化 | 已有验证类型 | 差异 |
| --- | --- | --- | --- |
| 01 | 批量分页 Decode | 完整服务与分页算子对照 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/main...opt/01-batch-paged-decode) |
| 02 | 扩大 Prefill query 分块 | 完整服务、Prefill 算子耗时和显存对照 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/01-batch-paged-decode...opt/02-query-chunk-128) |
| 03 | IndexPool 有序选组与因果尾组 | 完整服务、选组顺序和有效位逐项核对 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/02-query-chunk-128...opt/03-ordered-indexpool) |
| 04 | 原生稀疏 Prefill 与混合步 Decode | 完整服务按这一组实现计分，保留算子与短测边界 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/03-ordered-indexpool...opt/04-native-sparse-prefill) |
| 05 | 批量 Decode IndexPool | 完整服务与选组一致性对照 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/04-native-sparse-prefill...opt/05-batched-indexpool-decode) |
| 06 | 批量 Prefill 选组与四头滑窗 | 完整服务按组合计分，分别检查选组与滑窗数值 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/05-batched-indexpool-decode...opt/06-prefill-selection-swa) |
| 07 | 稳定 ACLNN Host 缓存 | 两种启动历史短测与完整服务 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/06-prefill-selection-swa...opt/07-aclnn-host-cache) |
| 08 | 融合小批 Decode 算子并复用索引 | 完整服务按组合计分，子项有算子或短测证据 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/07-aclnn-host-cache...opt/08-fused-small-decode) |
| 09 | 小批 Decode 本地专家与 FP32 归并 | 完整服务、路由归属和加权求和验证 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/08-fused-small-decode...opt/09-local-expert-decode) |
| 10 | 融合 IndexPool 和 iHC，配对归并专家输出 | 完整服务按组合计分；记录浮点归并及 BF16 回写变化 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/09-local-expert-decode...opt/10-joint-expert-reduce) |
| 11 | 融合宽批 iHC 并复用稀疏 Prefill 准备 | 两种启动历史短测及两轮完整服务；另保留整模型生成差异 | [查看差异](https://github.com/jiaran-king/hy4-flash-a3-optimize/compare/opt/10-joint-expert-reduce...opt/11-wide-prefill-sparse-setup) |

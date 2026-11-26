# Hy4 Flash on Ascend A3

腾讯 Hy4 Flash BF16 在昇腾 A3 上的适配、逐项性能优化与复现。参考场景为单节点 16 DIE、TP16 + EP、16,384 输入 / 1,024 输出 token、客户端并发 4。

- `main` 提供原始 A3 适配、部署和共用评测入口。
- `optimized` 提供按实测演进顺序叠加的完整补丁链。
- [逐阶段差异](docs/optimization.md) 用相邻版本的 Git 差异展示每项优化和依赖。

## 复现

使用 A3 容器、Python 3.12 和 Git，在本地挂载腾讯发布的 BF16 checkpoint。部署脚本会核对固定上游源码、A3 适配文件、原生扩展与模型配置和索引。运行环境和两版命令见 [部署与压测](docs/benchmark.md)。

```bash
# CPU 上检查基线源码是否可重建
python3 -m deploy.prepare_runtime --profile baseline --source-only --output runs/baseline

# 叠加并检查完整优化链
git switch optimized
python3 -m deploy.prepare_runtime --profile optimized --source-only --output runs/optimized
```

[本地结果格式与离线复算](results/README.md) · [GSM8K 与 RULER](docs/accuracy.md) · [来源与许可](NOTICE.md)

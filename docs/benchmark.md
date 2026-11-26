# 部署与压测

## 运行条件

用 Linux AArch64 的 A3 运行时，单节点 16 DIE。参考镜像标签、CANN、PyTorch、torch-npu 和原生扩展 SHA-256 见 [运行时配置](../configs/runtime.json)。需要从可用的 A3 镜像提供相符的扩展和算子库。用 `--native-root` 指向镜像内包含原生扩展、`_cann_ops_custom/vendors` 及 `lib` 的 `vllm_ascend` 包目录。CANN 路径可通过 `--cann-root` 指定。

模型使用 [tencent/Hy4-flash BF16](https://huggingface.co/tencent/Hy4-flash/tree/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959)。将 checkpoint、tokenizer 和 chat template 放在同一模型目录。部署核对 config、safetensors index、tokenizer 和 chat template 的内容哈希，以及 148 个索引分片的存在性。完整分片的内容校验由权重分发方提供。

## 启动

从仓库根目录执行。准备基线时可留在 `main`，完整候选使用 `git switch optimized`。使用阶段分支和 `--through <stage>` 可构建某个中间版本。源码包从 [固定版本](../configs/runtime.json) 下载，也可通过 `--archives` 复用同字节的下载。

```bash
python3 -m deploy.prepare_runtime --profile baseline --native-root "$HY4_A3_NATIVE_ROOT" \
  --output runs/baseline-a3
python3 -m deploy.launch_node --runtime runs/baseline-a3 --model-dir "$HY4_MODEL_DIR"
```

优化版使用同一启动命令，把准备阶段的 `--profile` 设为 `optimized` 并指定独立输出目录。两版共享 [服务和负载配置](../configs/workload.json)。默认只监听 localhost。使用 `--dry-run` 可在 CPU 上检查已准备源码的启动参数。

## 压测和复算

服务加载完成后，在另一终端执行。

```bash
python3 -m bench.run_case --runtime runs/baseline-a3 --model-dir "$HY4_MODEL_DIR" \
  --history baseline --output runs/measurements/baseline-c4
```

最终优化版用 `--history primed`，依次发一条 16K/16、四条 16K/16、四条 16K/128，再插一条 16K 请求到三个活跃 Decode 流中，最后跑七道文本质检和正式测量。基线的 `baseline` 历史在计分前跑七道质检。对比中间版本时记录实际预热历史。复测可复用已加载的服务，使用 `--history none` 并记录此前请求历史。已有同口径归档成绩时可直接离线复算对照。

正式负载为九条 16,384/1,024、并发 4、请求结束后补位，固定 seed、`PYTHONHASHSEED` 和温度，使用 vLLM 的 `timed_trace`。输出长短筛选可以指定 `--output-tokens 16` 或 `128`。完整流水件和本地回算见 [结果说明](../results/README.md)。

稳态窗口采用 DSV4.1 Case 23 的补位规则。窗口从并发首次补满开始，到最后一次满并发的时间点。吞吐按窗内实际发出的输出 token 数除以窗口秒数，单位为整节点 token/s。TTFT 与 TPOT 的稳态分位数来自窗口内发起的请求。首批最慢 TTFT 取客户端最早发出的四条请求，ITL 则从完整逐 token 流统计。

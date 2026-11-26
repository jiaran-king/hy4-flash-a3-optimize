# 精度评测

GSM8K 使用固定 300 题，按原快筛 128 题和后续补测顺序的前 172 题依次评估。复用 `gsm8k_cot` 的八个示例、停词和两种答案抽取。RULER 选 `niah_multivalue`、`vt`、`cwe` 各 32 题，输入固定为 16,384 token，按每题的全部参考答案计分。两套输入均校验已冻结的 tokenized JSONL 内容哈希和题号。归档的同前缀生成比较单独核对分歧 token 与 logits。

## 使用已归档的输入

```bash
python3 -m bench.accuracy.evaluate gsm8k --manifest "$HY4_GSM8K_JSONL" \
  --output runs/measurements/gsm8k-responses.jsonl
python3 -m bench.accuracy.evaluate ruler --manifest "$HY4_RULER_JSONL" \
  --output runs/measurements/ruler-responses.jsonl
```

两条命令连接当前已加载的服务，使用 [固定生成配置](../bench/accuracy/protocol.json)。已有整套回复时用 `--responses <jsonl>` 离线复算，按题号核对覆盖、token 数及终止状态。

## 重建输入

将 [协议](../bench/accuracy/protocol.json) 复制到工作目录的 `protocol.json`。在该目录按以下布局准备已固定的公开来源。论文原料和 tokenized 输入留在本地。

- `upstream/gsm8k/test.jsonl` 取 OpenAI GSM8K 的固定测试集。 [数据源码](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
- `upstream/lm-eval/gsm8k-cot.yaml` 取 EleutherAI 固定的任务定义。 [任务定义](https://github.com/EleutherAI/lm-evaluation-harness/blob/27988a293647d5853e48edea291640b4af54740c/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)
- `upstream/ruler/RULER-<revision>/` 解压 NVIDIA RULER 固定源码，按该项目安装依赖。将实验所用的 `PaulGrahamEssays.json` 放在 `upstream/`。参考文本内容摘要 SHA-256 为 `31775eb938cbc48d547ff645beb6713842ac9322866f61fd297f0ec0326a5bb0`。 [RULER 固定版本](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a)

使用匹配的 Hy4 tokenizer、chat template 和 Transformers 环境生成。仓库中的 [输入配置](../bench/accuracy/inputs.json) 记录期望的题集指纹。

```bash
python3 -m bench.accuracy.prepare_inputs gsm8k "$HY4_INPUT_DIR" "$HY4_MODEL_DIR"
python3 -m bench.accuracy.prepare_inputs ruler "$HY4_INPUT_DIR" "$HY4_MODEL_DIR"
```

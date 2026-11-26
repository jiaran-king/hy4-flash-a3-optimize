# 本地结果与复算

压测输出目录保留 vLLM 的 `raw.json`、合成请求轨迹、服务指标采样和带运行时身份的 `result.json`。记录中的 `itls` 和 `start_times` 可还原逐 token 时间线。原始请求、回复与完整运行记录应通过可访问的实验归档核对。这里保留公开复算入口，正式成绩由单独的验收记录引用。

```bash
python3 -m bench.report path/to/baseline/c4 path/to/optimized/c4
```

CPU 即可复算，目录中的固定负载长度和完成数必须匹配。每轮单列窗口内实际输出吞吐、稳态延迟、首批最慢 TTFT 以及 ITL。短测用 `--output-tokens 16` 或 `128`。未形成稳态的短测只读首批和全程计时。

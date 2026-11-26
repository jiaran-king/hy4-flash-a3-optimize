# Original A3 port

Start with the two checked source archives in [runtime.json](../../configs/runtime.json), then apply the recorded Ascend A5 integration. The local A3 port uses BF16 paged attention with FP32 accumulation and preserves the trained attention sinks. The IndexPool helper retains E4M3FN rounding using operations supported by A3. This baseline keeps `QUERY_CHUNK = 16`.

The common 16K/1K launch settings in [workload.json](../../configs/workload.json) use this unchanged adaptation. Use the [preparation command](../../docs/benchmark.md) to check the four archived port file digests.

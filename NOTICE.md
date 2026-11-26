# Source and license

The repository contains an A3 overlay, optimization diffs, and benchmark tools. Source distributions keep the upstream Apache-2.0 notices and authorship. See [LICENSE](LICENSE).

- vLLM HY4 support comes from the [pinned model integration](https://github.com/vllm-project/vllm/pull/68296).
- vLLM-Ascend's pinned base and linked integration commit come from the [upstream 950DT adaptation](https://github.com/vllm-project/vllm-ascend/pull/22540). The original integration patch and attribution are preserved in [patches/a3-baseline](patches/a3-baseline/).
- The checkpoint is supplied separately by [Tencent Hy](https://huggingface.co/tencent/Hy4-flash) under its own terms. Runtime libraries and benchmark data retain their upstream terms.

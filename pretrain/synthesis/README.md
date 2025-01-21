# Data Synthesis

This directory contains the scripts and prompts for data synthesis.

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

## Preliminary

### SGLang

We primarily use the [`sglang`](https://docs.sglang.ai/start/install.html) package to generate synthetic data.

Then, choose the model you want to use for data synthesis. For example, we use `DeepSeek-Prover-V1.5` and `Qwen2.5-Math-Instruct-7B` to augument the Lean theorem proving dataset.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-Prover-V1.5-RL --port 30000 --trust-remote-code --dp 2
```

For those who run the model on a large cluster, you can install the [`sglang_router`](https://docs.sglang.ai/router/router.html) package to optimize the data parallel scheduling efficiency.

```bash
pip install sglang-router
```

### vLLM

We also use the [`vLLM`](https://docs.vllm.ai/) package to generate synthetic data (on Ascend 910B NPU).

```bash
python gen_vllm.py --input_file_path input.jsonl --output_file_path output.jsonl
```

## Prompts

We have publish the prompts we used for data synthesis in our technical report. We will organize the synthesis pipeline soon.

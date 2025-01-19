# Data Synthesis

## Preparation

We primarily use the `sglang` package to generate synthetic data. To install the package, run the following command:

```bash
pip install sglang
```

Then, choose the model you want to use for data synthesis. For example, we use `DeepSeek-Prover-V1.5` and `Qwen2.5-Math-Instruct-7B` to augument the Lean theorem proving dataset.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-Prover-V1.5-RL --port 30000 --trust-remote-code --dp 2
```

## Prompts

We have publish the prompts we used for data synthesis in our technical report. We will organize the synthesis pipeline soon.

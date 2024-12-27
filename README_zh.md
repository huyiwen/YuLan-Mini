<h4 align="center">
    <p>
        <b>中文</b> | <a href="https://github.com/RUC-GSAI/YuLan-Mini">English</a> | <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/README_ja.md">日本語</a>
    <p>
</h4>

<div align=center>
<img src="assets/YuLan-logo.jpg" width="400px">
<h1>YuLan-Mini: 数据高效的开源语言模型</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2412.17743" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a><img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-Mini"></a>
</div>

YuLan-Mini 是一个 2.4B 参数量的轻量化语言模型。仅使用 1.08T Tokens 进行预训练，却达到了与使用更多数据的行业领先模型相媲美的性能，尤其是 **数学** 和 **代码** 两个领域。为方便复现，我们将开源相关预训练资源。

---

## 模型下载 🔗

> 我们将在整理后上传模型权重

|  模型  | 上下文长度 |
|--------|-------|
|  [YuLan-Mini-2.4B](https://huggingface.co/yulan-team/YuLan-Mini)（推荐）  |  28K |
|  [YuLan-Mini-2.4B-4k](https://huggingface.co/yulan-team/YuLan-Mini-Intermediate-4K) |  4K |

---

## 能力介绍 🌟

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/main.png">
</div>

我们的预训练方法通过以下三项关键技术改进提升了训练效率：

1. 精细的数据处理流程，将数据清洗与数据课程策略相结合；
2. 稳定的优化方法，有效缓解预训练中的不稳定性；
3. 高效的退火策略，融合了目标数据选择和长上下文训练。

最终，使用我们的高效预训练策略，仅 1T 的数据量便可在数学和代码等领域，媲美 Qwen2.5-1.5B 在 18T 数据上的效果。我们将开源使用到的 1T 数据，其中指令数据仅占 3.5%。

---
## 基准测试 🌟

|      Models      | Model Size | # Train Tokens | Context Length | MATH 500 | GSM 8K | Human Eval | MBPP   | RACE Middle | RACE High | RULER  |
|:----------------|----------:|--------------:|--------------:|:--------|:------|:----------|:------|:-----------|:---------|:------|
|     MiniCPM      |    2.6B    |     1.06T      |       4K       |   15.00  |  53.83 |     50.00* |  47.31 |     56.61   |   44.27   |   N/A  |
|      Qwen-2      |    1.5B    |       7T       |      128K      |   22.60  | 46.90* |     34.80* | 46.90* |     55.77   |   43.69   |  60.16 |
|     Qwen2.5      |    0.5B    |      18T       |      128K      |   23.60  | 41.60* |     30.50* | 39.30* |     52.36   |   40.31   |  49.23 |
|     Qwen2.5      |    1.5B    |      18T       |      128K      |   **45.40**  | **68.50\*** |     37.20* | 60.20* |     **58.77**   |   44.33   |  <ins>68.26</ins> |
|     Gemma2       |    2.6B    |       2T       |       8K       |   18.30* | 30.30* |     19.50* | 42.10* |       -     |      -    |   N/A  |
|    StableLM2     |    1.7B    |       2T       |       4K       |     -    |  20.62 |      8.50* |  17.50 |     56.33   |   **45.06**   |   N/A  |
|    SmolLM2       |    1.7B    |      11T       |       8K       |   11.80  |    -   |     23.35  |  45.00 |     55.77   |   43.06   |   N/A  |
|    Llama3.2      |    3.2B    |       9T       |      128K      |    7.40  |    -   |     29.30  |  49.70 |     55.29   |   43.34   |  **77.06** |
|    YuLan-Mini    |    2.4B    |     1.04T      |       4K       |   32.60  |  66.65 |     <ins>61.60</ins>  |  **66.70** |     55.71   |   43.58   |   N/A  |
|    YuLan-Mini    |    2.4B    |     1.08T      |      28K       |  <ins>37.80</ins>  |  <ins>68.46</ins> |    **64.00**  |  <ins>65.90</ins>|     <ins>57.18</ins>   |   <ins>44.57</ins>   |  51.48 |


|      Models      | LAMBADA | MMLU  | CMMLU | CEval | HellaSwag | WinoGrande | StoryCloze | ARC-e | ARC-c |
|:----------------|:-------|:-----|:-----|:-----|:----------|:-----------|:-----------|:-----|:-----|
|   MiniCPM-2.6B   |  61.91  | 53.37 | 48.97 | 48.24 |   67.92    |     65.74   |     78.51   | 55.51 | 43.86 |
|   Qwen2-1.5B     |  64.68  | 55.90 | **70.76** | **71.94** |   66.11    |     66.14   |     77.60   | 62.21 | 42.92 |
|  Qwen2.5-0.5B    |  52.00  | 47.50 | 52.17 | 54.27 |   50.54    |     55.88   |     71.67   | 56.10 | 39.51 |
|  Qwen2.5-1.5B    |  62.12  | <ins>60.71</ins> | <ins>67.82</ins> | <ins>69.05</ins> |   67.18    |     64.48   |     76.80   | **71.51** | <ins>53.41</ins> |
|   Gemma2-2.6B    |    -    | 52.20*|   -   | 28.00*|   <ins>74.60*</ins>   |    **71.50\***   |       -     |   -   | **55.70\***|
| StableLM2-1.7B   |  66.15  | 40.37 | 29.29 | 26.99 |   69.79    |     64.64   |     <ins>78.56</ins>   | 54.00 | 40.78 |
|  SmolLM2-1.7B    |  <ins>67.42</ins>  | 51.91 | 33.46 | 35.10 |   72.96    |     67.40   |     **79.32**   | 44.82 | 35.49 |
|   Llama3.2-3B    |  **69.08**  | **63.40** | 44.44 | 44.49 |   **75.62**    |     <ins>67.48</ins>   |     76.80   | <ins>70.12</ins> | 48.81 |
|    YuLan-Mini    |  64.72  | 51.79 | 48.35 | 51.47 |   68.65    |     67.09   |     76.37   | 69.87 | 50.51 |
|    YuLan-Mini    |  65.67  | 49.10 | 45.45 | 48.23 |   67.22    |     67.24   |     75.89   | 67.47 | 49.32 |

---

# 预训练资源

为了提升研究的透明度和可复现性，我们将开源预训练相关资源：


<details><summary>1. 预训练和评测代码</summary>

预训练和评测代码将在后续公布。
</details>

<details><summary>2. 中间阶段检查点</summary>

中间阶段的检查点发布在 <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini</a>。
</details>


<details><summary>3. 退火前优化器状态</summary>

退火前优化器状态将在后续公布。
</details>

<details><summary>4. 使用的开源数据集</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets-list.md">使用的开源数据集</a>

</details>

<details><summary>5. 逐阶段的数据配比</summary>


<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/final.pdf">数据配比高清图</a>
<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data_distribution_for_every_phase.png">
</div>
</details>


<details><summary>6. 合成数据</summary>

数据清洗和合成流程：
<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

我们所使用的合成数据发布在 <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini-Datasets</a>

</details>
<details><summary>7. 中间阶段优化器状态</summary>

中间阶段优化器状态将在后续公布。
</details>



---

## 快速开始 💻

以下是使用 Huggingface 的简单推理代码示例：

**Huggingface 推理示例**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("yulan-team/YuLan-Mini")
model = AutoModelForCausalLM.from_pretrained("yulan-team/YuLan-Mini", torch_dtype=torch.bfloat16)

# 输入文本
input_text = "Renmin University of China is"
inputs = tokenizer(input_text, return_tensors="pt")

# 推理
output = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```


---

## 许可协议

- 本仓库代码使用 [MIT License](./LICENSE)。
- 关于模型权重、中间优化器状态和训练数据的使用政策将在后续公布。
- 局限性：尽管我们尝试减少模型在使用中可能出现的安全性问题，并鼓励模型生成符合道德和法律要求的文本，但由于语言模型基于概率生成的范式，模型仍然可能会产生意外的输出。例如，生成的响应可能包含偏见、歧视或其他有害内容。请不要传播此类内容。我们对因传播有害信息而造成的任何后果不承担任何责任。

## 引用

如果您发现 YuLan-Mini 对您的研究或开发有帮助，请引用我们的[技术报告](https://arxiv.org/abs/2412.17743)：

```BibTex
@article{yulanmini,
      title        = {YuLan-Mini: An Open Data-efficient Language Model}, 
      author       = {Yiwen Hu and
                      Huatong Song and
                      Jia Deng and
                      Jiapeng Wang and
                      Jie Chen and
                      Kun Zhou and
                      Yutao Zhu and
                      Jinhao Jiang and
                      Zican Dong and
                      Wayne Xin Zhao and
                      Ji-Rong Wen},
      url          = {https://arxiv.org/abs/2412.17743},
      journal      = {CoRR},
      volume       = {abs/2412.17743},
      year         = {2024},
      url          = {https://doi.org/10.48550/arXiv.2412.17743},
      doi          = {10.48550/ARXIV.2412.17743},
      eprinttype   = {arXiv},
      eprint       = {2412.17743}
}
```

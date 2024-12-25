<h4 align="center">
    <p>
        <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/README_zh.md">中文</a> | <b>English</b>
    <p>
</h4>

<div align=center>
<img src="assets/YuLan-logo.jpg" width="400px">
<h1>YuLan-Mini: An Open Data-efficient Language Model</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2412.17743" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/yulan-team/YuLan-Mini"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a><img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-Mini"></a>
</div>

YuLan-Mini is a lightweight language model with 2.42 billion parameters. It was pre-trained using only 1.08T tokens yet achieves performance comparable to industry-leading models trained with significantly more data. To facilitate reproducibility, we are open-sourcing the relevant pre-training resources.

---

#### Model Downloads 🔗

> Model weights will be uploaded after final preparations.

|  Model  | Context Length |
|---------|----------------|
|  [YuLan-Mini-2.4B](https://huggingface.co/rucaibox/YuLan-Mini-2.4B) (Recommended)  |  28K |
|  [YuLan-Mini-2.4B-4K](https://huggingface.co/rucaibox/YuLan-Mini-2.4B-4K)  |  4K |

---

#### Features 🌟

<div align=center>
<img src="assets/main.png">
</div>

Our pre-training methodology improves training efficiency through three key innovations:

1. an elaborately designed **data pipeline** that combines data cleaning with data schedule strategies;
2. a systematic **optimization method** that can effectively mitigate training instability;
3. an effective **annealing approach** that integrate targeted data selection and long context training.


---
#### Behchmarks 🌟

|      Models      | Model Size | # Train Tokens | Context Length | MATH 500 | GSM 8K | Human Eval | MBPP   | RACE Middle | RACE High | RULER  |
|:----------------:|:----------:|:--------------:|:--------------:|:--------:|:------:|:----------:|:------:|:-----------:|:---------:|:------:|
|     MiniCPM      |    2.6B    |     1.06T      |       4K       |   15.00  |  53.83 |     50.00* |  47.31 |     56.61   |   44.27   |   N/A  |
|      Qwen-2      |    1.5B    |       7T       |      128K      |   22.60  | 46.90* |     34.80* | 46.90* |     55.77   |   43.69   |  60.16 |
|     Qwen2.5      |    0.5B    |      18T       |      128K      |   23.60  | 41.60* |     30.50* | 39.30* |     52.36   |   40.31   |  49.23 |
|     Qwen2.5      |    1.5B    |      18T       |      128K      |   45.40  | 68.50* |     37.20* | 60.20* |     58.77   |   44.33   |  68.26 |
|     Gemma2       |    2.6B    |       2T       |       8K       |   18.30* | 30.30* |     19.50* | 42.10* |       -     |      -    |   N/A  |
|    StableLM2     |    1.7B    |       2T       |       4K       |     -    |  20.62 |      8.50* |  17.50 |     56.33   |   45.06   |   N/A  |
|    SmolLM2       |    1.7B    |      11T       |       8K       |   11.80  |    -   |     23.35  |  45.00 |     55.77   |   43.06   |   N/A  |
|    Llama3.2      |    3.2B    |       9T       |      128K      |    7.40  |    -   |     29.30  |  49.70 |     55.29   |   43.34   |  77.06 |
|    YuLan-Mini    |    2.4B    |     1.04T      |       4K       |   32.60  |  66.65 |     61.60  |  66.70 |     55.71   |   43.58   |   N/A  |
|    YuLan-Mini    |    2.4B    |     1.08T      |      28K       |   37.80  |  68.46 |     64.00  |  65.90 |     57.18   |   44.57   |  51.48 |


|      Models      | LAMBADA | MMLU  | CMMLU | CEval | HellaSwag | WinoGrande | StoryCloze | ARC-e | ARC-c |
|:----------------:|:-------:|:-----:|:-----:|:-----:|:----------:|:-----------:|:-----------:|:-----:|:-----:|
|   MiniCPM-2.6B   |  61.91  | 53.37 | 48.97 | 48.24 |   67.92    |     65.74   |     78.51   | 55.51 | 43.86 |
|   Qwen2-1.5B     |  64.68  | 55.90 | 70.76 | 71.94 |   66.11    |     66.14   |     77.60   | 62.21 | 42.92 |
|  Qwen2.5-0.5B    |  52.00  | 47.50 | 52.17 | 54.27 |   50.54    |     55.88   |     71.67   | 56.10 | 39.51 |
|  Qwen2.5-1.5B    |  62.12  | 60.71 | 67.82 | 69.05 |   67.18    |     64.48   |     76.80   | 71.51 | 53.41 |
|   Gemma2-2.6B    |    -    | 52.20*|   -   | 28.00*|   74.60*   |    71.50*   |       -     |   -   | 55.70*|
| StableLM2-1.7B   |  66.15  | 40.37 | 29.29 | 26.99 |   69.79    |     64.64   |     78.56   | 54.00 | 40.78 |
|  SmolLM2-1.7B    |  67.42  | 51.91 | 33.46 | 35.10 |   72.96    |     67.40   |     79.32   | 44.82 | 35.49 |
|   Llama3.2-3B    |  69.08  | 63.40 | 44.44 | 44.49 |   75.62    |     67.48   |     76.80   | 70.12 | 48.81 |
|    YuLan-Mini    |  64.72  | 51.79 | 48.35 | 51.47 |   68.65    |     67.09   |     76.37   | 69.87 | 50.51 |
|    YuLan-Mini    |  65.67  | 49.10 | 45.45 | 48.23 |   67.22    |     67.24   |     75.89   | 67.47 | 49.32 |

---

#### Inference Code 💻

Below is a simple example for inference using Huggingface:

**Huggingface Inference Example**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("rucaibox/YuLan-Mini-2.4B")
model = AutoModelForCausalLM.from_pretrained("rucaibox/YuLan-Mini-2.4B")

# Input text
input_text = "Renmin University of China is"
inputs = tokenizer(input_text, return_tensors="pt")

# Completion
output = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

#### Pre-Training Resources 🔧

To enhance research transparency and reproducibility, we are open-sourcing relevant [pre-training resources](https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain):

<details><summary>1. Pre-training and Evaluation Code</summary>

The pre-training and evaluation code will be released in a future update.
</details>



<details><summary>2. Intermediate Stage Checkpoints</summary>
The intermediate stage checkpoints will be released in a future update.

</details>

<details><summary>3. Optimizer States Before Annealing</summary>

Optimizer states before annealing will be released in a future update.
</details>

<details><summary>3. Stage-wise Data Ratios</summary>

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-preview.png">
</div>
</details>

<details><summary>4. The Open-Source Datasets /summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets-list.md">Used-Datasets-List</a>

</details>

<details><summary>5. Data Distribution for every phase</summary>


<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/final.pdf">High-Definition Image</a>

</details>

<details><summary>6. Synthetic Data</summary>

Data cleaning and synthesis pipeline:
<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>
</details>

<details><summary>7. Intermediate Optimizer States</summary>

Intermediate optimizer states will be released in a future update.
</details>




---

## License

- The code in this repository is released under the [MIT License](./LICENSE).
- Policies regarding the use of model weights, intermediate optimizer states, and training data will be announced in future updates.
- Limitations: Despite our efforts to mitigate safety concerns and encourage the generation of ethical and lawful text, the probabilistic nature of language models may still lead to unexpected outputs. For instance, responses might contain bias, discrimination, or other harmful content. Please refrain from disseminating such content. We are not liable for any consequences arising from the spread of harmful information.

### Citation

If you find YuLan-Mini helpful for your research or development, please cite [our technical report](https://arxiv.org/abs/2412.17743):

```
@misc{hu2024yulanmini,
      title={YuLan-Mini: An Open Data-efficient Language Model}, 
      author={Yiwen Hu and Huatong Song and Jia Deng and Jiapeng Wang and Jie Chen and Kun Zhou and Yutao Zhu and Jinhao Jiang and Zican Dong and Wayne Xin Zhao and Ji-Rong Wen},
      year={2024},
      eprint={2412.17743},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17743}, 
}
```

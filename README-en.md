<h4 align="center">
    <p>
        <a href="https://github.com/RUC-GSAI/YuLan-Mini">中文</a> | <b>English</b>
    <p>
</h4>

<div align=center>
<img src="assets/YuLan-logo.jpg" width="400px">
<h1>YuLan-Mini: An Open Data-efficient Language Model</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/rucaibox"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
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

<details><summary>2. Optimizer States Before Annealing</summary>

Optimizer states before annealing will be released in a future update.
</details>

<details><summary>3. Stage-wise Data Ratios</summary>

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-preview.png">
</div>
</details>

<details><summary>4. Synthetic Data</summary>

Data cleaning and synthesis pipeline:
<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>
</details>

<details><summary>5. Intermediate Optimizer States</summary>

Intermediate optimizer states will be released in a future update.
</details>

---

## Team

YuLan-Mini is developed by the [AI Box](http://aibox.ruc.edu.cn/) team at Renmin University of China.

## License

- The code in this repository is released under the [MIT License](./LICENSE).
- Policies regarding the use of model weights, intermediate optimizer states, and training data will be announced in future updates.
- Limitations: Despite our efforts to mitigate safety concerns and encourage the generation of ethical and lawful text, the probabilistic nature of language models may still lead to unexpected outputs. For instance, responses might contain bias, discrimination, or other harmful content. Please refrain from disseminating such content. We are not liable for any consequences arising from the spread of harmful information.

### Citation

If you find YuLan-Mini helpful for your research or development, please cite [our technical report]():

```
```
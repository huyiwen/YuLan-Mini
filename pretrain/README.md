# Pre-Training Resources 🔧

To enhance research transparency and reproducibility, we are open-sourcing relevant pre-training resources:

<details><summary>1. Pre-training and Evaluation Code</summary>

The pre-training and evaluation code will be released in a future update.
</details>



<details><summary>2. Intermediate Stage Checkpoints</summary>
The intermediate stage checkpoints are released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini</a>.

<table>
    <thead>
        <tr>
            <th>Stage</th>
            <th>Curriculum Phase</th>
            <th>4K Context</th>
            <th>28K Context</th>
            <th>Optimizer</th>
            <th>Inference Arch</th>
            <th>LAMBADA `Acc`</th>
            <th>GSM8K `Acc`</th>
            <th>HumanEval `pass@1`</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Stable</td>
            <td>5</td>
            <td>[YuLan-Mini-Phase5](https://huggingface.co/yulan-team/YuLan-Mini-Phase5)</td>
            <td></td>
            <td></td>
            <td>`yulanmini`</td>
            <td>53.85</td>
            <td>3.41</td>
            <td>12.26</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>10</td>
            <td>[YuLan-Mini-Phase10](https://huggingface.co/yulan-team/YuLan-Mini-Phase10)</td>
            <td></td>
            <td></td>
            <td>`yulanmini`</td>
            <td>55.00</td>
            <td>9.57</td>
            <td>15.95</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>15</td>
            <td>[YuLan-Mini-Phase15](https://huggingface.co/yulan-team/YuLan-Mini-Phase15)</td>
            <td></td>
            <td></td>
            <td>`yulanmini`</td>
            <td>55.81</td>
            <td>13.81</td>
            <td>16.99</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>20</td>
            <td>[YuLan-Mini-Phase20](https://huggingface.co/yulan-team/YuLan-Mini-Phase20)</td>
            <td></td>
            <td>✅</td>
            <td>`yulanmini`</td>
            <td>55.81</td>
            <td>21.39</td>
            <td>20.79</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>25 (1T tokens)</td>
            <td>[YuLan-Mini-Before-Annealing](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing)</td>
            <td></td>
            <td>✅</td>
            <td>`yulanmini`</td>
            <td>55.67</td>
            <td>29.94</td>
            <td>34.06</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>26</td>
            <td>YuLan-Mini-4K</td>
            <td></td>
            <td></td>
            <td>`llama`\*</td>
            <td>64.72</td>
            <td>66.65</td>
            <td>61.60</td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>27</td>
            <td></td>
            <td>[YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini)</td>
            <td></td>
            <td>`llama`\*</td>
            <td>65.67</td>
            <td>68.46</td>
            <td>64.00</td>
        </tr>
    </tbody>
</table>

</details>

<details><summary>3. Optimizer States Before Annealing</summary>

<a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">YuLan-Mini-Before-Annealing</a>
</details>


<details><summary>4. The Used Open-Source Datasets </summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets">Used-Datasets-List</a>

</details>

<details><summary>5. Data Distribution for every phase</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets/final.pdf">
  <div align=center>
    <img src="assets/data_distribution_for_every_phase.png">
  </div>
</a>

</details>

<details><summary>6. Synthetic Data</summary>

Data cleaning and synthesis pipeline:
<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

The synthetic data we are using is released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini-Datasets</a>

</details>


### What you can do with these pre-training resources

1. **Pre-train** your own LLM. You can use [our data](https://huggingface.co/yulan-team/YuLan-Mini-Datasets) and curriculum to train a model that's just as powerful as YuLan-Mini.
2. Perform your own **learning rate annealing**. During the annealing phase, YuLan-Mini's learning ability is at its peak. You can resume training from [the checkpoint before annealing](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing) and use your own dataset for learning rate annealing.
3. **Fine-tune** the Instruct version of the LLM. You can use the [YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini) base model to train your own Instruct version.
4. **Training dynamics** research. You can use YuLan-Mini's [intermediate checkpoints](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3) to explore internal changes during the pre-training process.
5. **Synthesize** your own data. You can use YuLan-Mini's [data pipeline](https://github.com/RUC-GSAI/YuLan-Mini) to clean and generate your own dataset.


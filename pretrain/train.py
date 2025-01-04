import json
import math
import os

from tqdm import tqdm

print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
print(f"RANK: {os.environ['RANK']}")
print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

import copy
import glob
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.distributed as dist
import transformers
import wandb
from datasets import disable_caching
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed
from transformers.integrations.deepspeed import deepspeed_config

from train_utils import LogCallback, PyTorchProfilerCallback, print_rank0
from yulanmini_trainer import YuLanMiniTrainer

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    flash_attention: Optional[bool] = field(default=False)

    # deprecated: embedding gradient shrink
    shrink_alpha: float = field(
        default=1, metadata={"help": "Shrink alpha for shrink embedding grad"})
    shrink_alpha2: float = field(
        default=1, metadata={"help": "Shrink alpha for shrink embedding grad"})

    # deprecated: Pre-LN or Post-LN
    pre_ln: bool = field(default=True, metadata={"help": "Use pre layer norm"})
    post_ln: bool = field(default=False,
                          metadata={"help": "Use post layer norm"})

    # deprecated: embedding layer norm
    embedding_ln: bool = field(default=False,
                               metadata={"help": "Use embedding layer norm"})
    embedding_rmsln: bool = field(
        default=False, metadata={"help": "Use embedding rms layer norm"})

    # deprecated: scaling factor for LN
    ln_scale: Optional[float] = field(
        default=1., metadata={"help": "embedding_ln * ln_scale"})

    # build model from cli
    intermediate_size: Optional[int] = field(
        default=None, metadata={"help": "intermediate size"})
    hidden_size: Optional[int] = field(default=None,
                                       metadata={"help": "hidden size"})
    num_hidden_layers: Optional[int] = field(
        default=None, metadata={"help": "num_hidden_layers"})
    num_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "num_attention_heads"})
    num_key_value_heads: Optional[int] = field(
        default=None, metadata={"help": "num_key_value_heads"})
    initializer_range: Optional[float] = field(
        default=None, metadata={"help": "initializer_range"})
    config_dtype: str = field(
        default="bfloat16",
        metadata={"help": "The dtype of the model configuration."})
    attention_bias: bool = field(default=True, metadata={"help": "qkv bias"})
    tie_word_embeddings: bool = field(default=False,
                                      metadata={"help": "tie_word_embeddings"})

    # scaled init
    init_scale_o: Optional[float] = field(
        default=None,
        metadata={"help": "init_scale_o，影响O和D的初始化倍数（Spike和Cerebras相同）"})

    # muP
    dim_model_base: Optional[float] = field(
        default=None, metadata={"help": "D_model,0, 影响logits放缩和学习率"})
    dim_ffn_base: Optional[float] = field(default=None,
                                          metadata={"help": "D_ffn,0"})
    dim_model_base_init: Optional[float] = field(
        default=None,
        metadata={
            "help":
            "D_model,0 可以是None（相当于不使用任何放缩，所有都初始化成initializer_range），也可以是float"
        }
    )  # math.sqrt(2*(self.config.hidden_size / self.config.dim_model_base_init)*self.config.num_hidden_layers) 或者 math.sqrt((self.config.hidden_size / self.config.dim_model_base_init))
    dim_ffn_base_init: Optional[float] = field(
        default=None, metadata={"help": "D_ffn,0,默认值是1"})
    dim_model_base_attn: Optional[float] = field(
        default=None,
        metadata={
            "help":
            "D_model,0，影响Attention的放缩，默认值None是使用transformers默认初始化，1是使用cerebras初始化"
        })
    dim_model_base_logits: Optional[float] = field(
        default=None, metadata={"help": "D_model,0 , logits的单独放缩"})
    dim_model_base_lr: Optional[float] = field(
        default=None, metadata={"help": "D_model,0 , lr的单独放缩"})
    vi_residual_alpha: float = field(
        default=None,
        metadata={
            "help":
            "The alpha for vi residual，影响residual放缩，在cerebras中没有使用，在minicpm和Tensor Program VI中使用"
        })
    use_muparam_lr: bool = field(
        default=False,
        metadata={
            "help":
            "Use muparam learning rate，使用muparam学习率放缩，在cerebras和minicpm中使用"
        })
    scale_emb: Optional[float] = field(default=1,
                                       metadata={"help": "scale_emb"})
    model_reproduce: str = field(
        default="cerebras",
        metadata={"help": "what variants of mup: minicpm, cerebras, or transformers"})

    # qk layernorm
    qk_layernorm: bool = field(default=False,
                               metadata={"help": "Use qk layernorm"})

    # z-loss
    z_loss: float = field(default=0.0, metadata={"help": "z_loss"})

    # initialize the wesar learnable parameters
    wesar_weights: bool = field(default=True,
                                metadata={"help": "use weasr alpha"})
    use_emb_alpha: bool = field(default=False,
                                metadata={"help": "enable wesar for embedding"})
    embed_tokens_alpha: float = field(default=1,
                                      metadata={"help": "embed_tokens_alpha"})
    q_proj_alpha: float = field(default=1, metadata={"help": "q_proj_alpha"})
    k_proj_alpha: float = field(default=1, metadata={"help": "k_proj_alpha"})
    v_proj_alpha: float = field(default=1, metadata={"help": "v_proj_alpha"})
    o_proj_alpha: float = field(default=1, metadata={"help": "o_proj_alpha"})
    down_proj_alpha: float = field(default=1,
                                   metadata={"help": "down_proj_alpha"})
    gate_up_proj_alpha: float = field(default=1,
                                      metadata={"help": "gate_up_proj_alpha"})
    use_norm_alpha: bool = field(default=True,
                                 metadata={"help": "enable wesar for layernorm (i.e. input_ln, post_attention_ln, and ln)"})
    input_layernorm_alpha: float = field(
        default=1, metadata={"help": "input_layernorm_alpha"})
    post_attention_layernorm_alpha: float = field(
        default=1, metadata={"help": "post_attention_layernorm_alpha"})
    norm_alpha: float = field(default=1, metadata={"help": "norm_alpha"})
    lm_head_alpha: float = field(default=1, metadata={"help": "lm_head_alpha"})

    # gradient checkpointing
    gradient_checkpointing_step: int = field(
        default=7, metadata={"help": "gradient_checkpointing_step"})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    log_dir: str = field(default=None)
    profile: bool = field(default=False)
    # wsd scheduler
    use_wsd: bool = field(default=True)
    start_lambda: float = field(
        default=0, metadata={"help": "LR scheduler start percent"})
    end_lambda: float = field(default=1,
                              metadata={"help": "LR scheduler end percent"})
    start_global_step: float = field(
        default=None, metadata={"help": "LR scheduler start step"})
    end_global_step: float = field(default=None,
                                   metadata={"help": "LR scheduler end step"})
    wsd_style: str = field(
        default="",
        metadata={"help": "The style of the WSD loss, cos or linear."})
    # curriculum learning
    add_rms_norm: bool = field(default=False,
                               metadata={"help": "Choose models with 'RMSNorm' when resuming."})
    update_trained_steps_and_epochs: bool = field(  # whether to start a new curriculum phase
        default=False,
        metadata={
            "help":
            "Update the trainer state with the trained steps and epochs."  # 每一轮开始时更新读取的step和epoch，否则从模型的config.json中读取
        })
    num_steps_trained_before_this_epoch: int = field(
        default=0,
        metadata={"help": "多少步在这个epoch之前训过"})  # /home/u20140041/pretrain-mini/.venv/lib/python3.12/site-packages/transformers/trainer.py:2168
    num_epochs_trained_before_this_epoch: int = field(
        default=0,
        metadata={"help": "多少个epoch在这个epoch之前训过"})
pass




class PretrainDataset(Dataset):

    # To use doc attention mask, add "position_ids" column to the `train_dataset`

    def __init__(self, train_dataset, eos_token_id, skip=0):
        super(PretrainDataset, self).__init__()
        self.sources = train_dataset
        self.banned_idx = set()  # <-- Add banned indices here (not used)
        self.available_idx = []
        self.sorted_banned = sorted(self.banned_idx)
        self.skip = skip
        self.size = len(self.sources) - len(self.banned_idx) + skip
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        idx -= self.skip
        try:
            f = self.sorted_banned.index(idx)
        except ValueError:
            f = None
        if f is not None:
            idx = self.available_idx[f]

        ipt_ids = self.sources[idx]["input_ids"]
        results = dict(input_ids=ipt_ids,
                       labels=copy.deepcopy(ipt_ids),
                       idx=idx)
        if "position_ids" in self.sources[idx]:  # for doc attention mask
            results["position_ids"] = self.sources[idx]["position_ids"]
        if "subset" in self.sources[idx]:  # record the subset
            results["subset"] = self.sources[idx]["subset"]

        return results


@dataclass
class DataCollatorForPretrainDataset:
    data_args: DataArguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        r = dict(
            input_ids=torch.tensor([d["input_ids"] for d in instances]),
            labels=torch.tensor([d["labels"] for d in instances]),
            subset=[d["subset"]
                    for d in instances] if "subset" in instances[0] else None,
            idx=[d["idx"] for d in instances],
        )
        if "position_ids" in instances[0]:  # for doc attention mask
            r["position_ids"] = torch.tensor(
                [d["position_ids"] for d in instances])
        return r


def prepare_data(tokenizer,
                                data_args,
                                training_args,
                                model_args,
                                skip=0):

    # staggered start
    sleep_time = RANK
    time.sleep(sleep_time / 2)

    # load the dataset
    train_dataset = []
    for data_name in sorted(os.listdir(data_args.data_path)):
        # support parquet format dataset to save disk space
        for load_fn in [
                lambda d: datasets.load_dataset(d, split="train", num_proc=8),
                datasets.load_from_disk,
        ]:
            try:
                d = load_fn(os.path.join(data_args.data_path, data_name))
                break
            except Exception as e:
                print(e, "trying next method")
                d = None
                continue
        if d is None:
            raise ValueError(f"Failed to load dataset {data_name}")

        d = d.add_column("subset", [data_name] * len(d))
        train_dataset.append(d)
        print(f"Dataset {data_name} loaded")

    # print the start index of each dataset
    if RANK == 0:
        start_index = 0
        for d in train_dataset:
            print(f"Dataset {d['subset'][0]} start index: {start_index}")
            start_index += len(d)

    print(len(train_dataset))
    train_dataset = datasets.concatenate_datasets(train_dataset)
    print(train_dataset)

    print(f"train dataset size: {len(train_dataset)}")
    if RANK == WORLD_SIZE - 1:
        for index in [0] + list(random.sample(range(len(train_dataset)), 20)):
            print(
                f"Sample {index} of the training set: {train_dataset[index]}.")
            print("---------" * 9)
            if isinstance(train_dataset[index]["input_ids"][0], list):
                print(tokenizer.decode(train_dataset[index]["input_ids"][0]))
            else:
                print(tokenizer.decode(train_dataset[index]["input_ids"]))
            print("=========" * 9)

    train_dataset = PretrainDataset(train_dataset=train_dataset,
                                      eos_token_id=tokenizer.eos_token_id,
                                      skip=skip)
    data_collator = DataCollatorForPretrainDataset(data_args=data_args,
                                                     tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def get_model_tokenizer(model_args, data_args, training_args):
    from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast

    from configuration_yulanmini import YuLanMiniConfig
    from modeling_yulanmini import YuLanMiniModelForCausalLM

    if not os.path.exists(model_args.model_name_or_path):

        checkpoint_zero = os.path.join(training_args.output_dir,
                                       "checkpoint-0")

        if RANK == 0:

            if model_args.model_name_or_path == "scratch-v1":  # divergent training
                # Note: Due to subsequent modifications to the training code, this model may require re-adaptation.

                config = YuLanMiniConfig(
                    vocab_size=102400,
                    hidden_size=1800,
                    intermediate_size=4800,
                    num_hidden_layers=56,
                    num_attention_heads=30,
                    num_key_value_heads=5,
                    max_position_embeddings=4096,
                    torch_dtype=getattr(torch, model_args.config_dtype),
                    tie_word_embeddings=True,
                    # hidden_act="silu",
                    step_scale=model_args.step_scale,
                )
                model = YuLanMiniModelForCausalLM(config)

                tokenizer = AutoTokenizer.from_pretrained(
                    "/home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731",
                    padding_side="right",
                )

            elif model_args.model_name_or_path == "scratch-v2":  # divergent training
                # Note: Due to subsequent modifications to the training code, this model may require re-adaptation.

                config = YuLanMiniConfig(
                    vocab_size=102400,
                    hidden_size=1800,
                    intermediate_size=9600,
                    num_hidden_layers=32,
                    num_attention_heads=12,
                    num_key_value_heads=2,
                    max_position_embeddings=4096,
                    torch_dtype=getattr(torch, model_args.config_dtype),
                    tie_word_embeddings=True,
                    # hidden_act="silu",
                    step_scale=model_args.step_scale,
                )
                model = YuLanMiniModelForCausalLM(config)

                tokenizer = AutoTokenizer.from_pretrained(
                    "/home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731",
                    padding_side="right",
                )

            elif model_args.model_name_or_path == "1.11-muparam":  # proxy model
                # Note: Due to subsequent modifications to the training code, this model may require re-adaptation.

                config = YuLanMiniConfig(
                    vocab_size=99000,
                    hidden_size=model_args.hidden_size,
                    intermediate_size=model_args.intermediate_size,
                    num_hidden_layers=model_args.num_hidden_layers,
                    num_attention_heads=model_args.num_attention_heads,
                    num_key_value_heads=model_args.num_key_value_heads,
                    max_position_embeddings=4096,
                    torch_dtype=getattr(torch, model_args.config_dtype),
                    tie_word_embeddings=model_args.tie_word_embeddings,
                    attention_bias=model_args.attention_bias,
                    dim_model_base=model_args.dim_model_base,
                    hidden_act="silu",
                    initializer_range=model_args.initializer_range,
                    shrink_alpha=1,  # not use in muparam
                    init_scale_o=1,  # not use in muparam
                    hidden_states_shrink=model_args.vi_residual_alpha /
                    math.sqrt(
                        model_args.num_hidden_layers),  # not use in muparam
                    dim_ffn_base_init=model_args.dim_ffn_base_init,
                    dim_model_base_init=model_args.dim_model_base_init,
                    dim_model_base_attn=model_args.dim_model_base_attn,
                    scale_emb=model_args.scale_emb,
                    model_reproduce=model_args.model_reproduce,
                )
                model = YuLanMiniModelForCausalLM(config)

                tokenizer = AutoTokenizer.from_pretrained(
                    "/home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731",
                    padding_side="right",
                )

            elif model_args.model_name_or_path == "cerebras":  # pure cerebras muP config
                # Note: Due to subsequent modifications to the training code, this model may require re-adaptation.

                config = YuLanMiniConfig(
                    vocab_size=99000,
                    hidden_size=model_args.hidden_size,
                    intermediate_size=model_args.intermediate_size,
                    num_hidden_layers=model_args.num_hidden_layers,
                    num_attention_heads=model_args.num_attention_heads,
                    num_key_value_heads=model_args.num_key_value_heads,
                    max_position_embeddings=4096,
                    torch_dtype=getattr(torch, model_args.config_dtype),
                    tie_word_embeddings=model_args.tie_word_embeddings,
                    attention_bias=model_args.attention_bias,
                    dim_model_base=model_args.dim_model_base,
                    hidden_act="silu",
                    initializer_range=model_args.initializer_range,
                    shrink_alpha=1,  # not use in muparam
                    init_scale_o=1,  # not use in muparam
                    hidden_states_shrink=model_args.vi_residual_alpha /
                    math.sqrt(
                        model_args.num_hidden_layers),  # not use in muparam
                    dim_ffn_base_init=model_args.dim_ffn_base_init,
                    dim_model_base_init=model_args.dim_model_base_init,
                    dim_model_base_attn=model_args.dim_model_base_attn,
                    scale_emb=model_args.scale_emb,
                    model_reproduce=model_args.model_reproduce,
                    embedding_ln=model_args.embedding_ln,
                    embedding_rmsln=model_args.embedding_rmsln,
                    ln_scale=model_args.ln_scale,
                    z_loss=model_args.z_loss,
                    gradient_checkpointing_step=model_args.
                    gradient_checkpointing_step,
                )
                model = YuLanMiniModelForCausalLM(config)

                tokenizer = AutoTokenizer.from_pretrained(
                    "/home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731",
                    padding_side="right",
                )

            elif model_args.model_name_or_path == "cerebras-wesar":  # cerebras muP + wesar

                config = YuLanMiniConfig(
                    vocab_size=99000,
                    hidden_size=model_args.hidden_size,
                    intermediate_size=model_args.intermediate_size,
                    num_hidden_layers=model_args.num_hidden_layers,
                    num_attention_heads=model_args.num_attention_heads,
                    num_key_value_heads=model_args.num_key_value_heads,
                    max_position_embeddings=4096,
                    torch_dtype=getattr(torch, model_args.config_dtype),
                    tie_word_embeddings=model_args.tie_word_embeddings,
                    attention_bias=model_args.attention_bias,
                    # dim_model_base=model_args.dim_model_base,
                    hidden_act="silu",
                    initializer_range=model_args.initializer_range,
                    shrink_alpha=1,  # not use in muparam
                    init_scale_o=1,  # not use in muparam
                    # hidden_states_shrink=1,  # wesar+纯cerebras是没有hidden_states_shrink的
                    hidden_states_shrink=model_args.vi_residual_alpha /
                    math.sqrt(
                        model_args.num_hidden_layers),  # not use in muparam
                    scale_emb=model_args.scale_emb,
                    model_reproduce=model_args.model_reproduce,  # fix
                    embedding_ln=model_args.embedding_ln,
                    embedding_rmsln=model_args.embedding_rmsln,
                    ln_scale=1,
                    z_loss=model_args.z_loss,
                    gradient_checkpointing_step=model_args.
                    gradient_checkpointing_step,
                    # wesar
                    embed_tokens_alpha=model_args.embed_tokens_alpha,
                    q_proj_alpha=model_args.q_proj_alpha,
                    k_proj_alpha=model_args.k_proj_alpha,
                    v_proj_alpha=model_args.v_proj_alpha,
                    o_proj_alpha=model_args.o_proj_alpha,
                    down_proj_alpha=model_args.down_proj_alpha,
                    gate_up_proj_alpha=model_args.gate_up_proj_alpha,
                    input_layernorm_alpha=model_args.input_layernorm_alpha,
                    post_attention_layernorm_alpha=model_args.
                    post_attention_layernorm_alpha,
                    norm_alpha=model_args.norm_alpha,
                    lm_head_alpha=model_args.lm_head_alpha,
                    dim_model_base_lr=model_args.dim_model_base_lr,
                    dim_model_base_logits=model_args.dim_model_base_logits,
                    wesar_weights=model_args.wesar_weights,
                    use_norm_alpha=model_args.use_norm_alpha,
                    use_emb_alpha=model_args.use_norm_alpha,
                    rms_type="llama",
                )
                model = YuLanMiniModelForCausalLM(config)

                tokenizer = AutoTokenizer.from_pretrained(
                    "/home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731",
                    padding_side="right",
                )
            else:
                raise ValueError()

            # Ensure the model is same across all ranks
            print(f"Saving model to {checkpoint_zero}")
            print(model)
            model.config.use_liger = True
            if os.path.exists(checkpoint_zero):
                shutil.rmtree(checkpoint_zero)
            os.makedirs(checkpoint_zero, exist_ok=True)
            model.save_pretrained(checkpoint_zero)
            tokenizer.save_pretrained(checkpoint_zero)

        dist.barrier()
        model_name_or_path = checkpoint_zero
    else:
        model_name_or_path = model_args.model_name_or_path

    # load model and tokenizer
    config = YuLanMiniConfig.from_pretrained(model_name_or_path)
    model = YuLanMiniModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2"
        if model_args.flash_attention else None,
        torch_dtype=getattr(torch, model_args.config_dtype),
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    if model_args.vi_residual_alpha is not None:
        if not math.isclose(
                model.config.hidden_states_shrink,
                model_args.vi_residual_alpha /
                math.sqrt(model.config.num_hidden_layers)):
            raise ValueError(
                f"hidden_states_shrink {model.config.hidden_states_shrink} is not equal to vi_residual_alpha {model_args.vi_residual_alpha} / math.sqrt(model.config.num_hidden_layers) {model.config.num_hidden_layers}"
            )

    if RANK == 0:
        print("params", sum(p.numel() for p in model.parameters()))

        print(model)
        print(model.model.embed_tokens.weight.data.norm())
        print(tokenizer)
        print(model.config)

    assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"
    return model, tokenizer


def train():
    wandb.login(key="xxx")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # automatically find the latest checkpoint to resume from. compatible with torchrun --max_restarts
    if not re.match(r".*/checkpoint-\d+", model_args.model_name_or_path):
        checkpoints = [
            c
            for c in glob.glob(f"{model_args.model_name_or_path}/checkpoint-*")
            if os.path.basename(c).split("-")[1].isdigit()
        ]
        if training_args.add_rms_norm:
            checkpoints = [c for c in checkpoints if c.endswith('-rms_norm')]
        else:
            checkpoints = [c for c in checkpoints if not c.endswith('-rms_norm')]
        if len(checkpoints) > 0:
            model_args.model_name_or_path = max(
                checkpoints, key=lambda x: int(os.path.basename(x).split("-")[1]))
            training_args.resume_from_checkpoint = model_args.model_name_or_path
        else:
            raise ValueError()

    # Re-balancing the matrix obtained from reparametrization and the learnable scaling factor. Performing this step did not significantly impact the results in the ablation study.
    if training_args.add_rms_norm and not model_args.model_name_or_path.endswith(
            "-rms_norm"):
        model_args.model_name_or_path += "-rms_norm"
        training_args.resume_from_checkpoint = model_args.model_name_or_path

    # Calculate where to resume training, i.e. new phase or same phase
    if training_args.num_steps_trained_before_this_epoch != 0 or training_args.num_epochs_trained_before_this_epoch != 0:
        raise ValueError(
            "num_steps_trained_before_this_epoch and num_epochs_trained_before_this_epoch 不应该从命令行传入")
    elif training_args.update_trained_steps_and_epochs:  # new curriculum phase
        with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")) as f:
            state = json.load(f)
        training_args.num_steps_trained_before_this_epoch  = state["global_step"]
        training_args.num_epochs_trained_before_this_epoch = math.ceil(state["epoch"])
        del state
    else:  # continue training in the same curriculum phase
        with open(os.path.join(training_args.resume_from_checkpoint, "config.json")) as f:
            config = json.load(f)
        training_args.num_steps_trained_before_this_epoch  = config["num_steps_trained_before_this_epoch"]
        training_args.num_epochs_trained_before_this_epoch = config["num_epochs_trained_before_this_epoch"]
        del config

    print("=="*50)
    print(training_args)
    print("=="*50)
    print(training_args.num_epochs_trained_before_this_epoch)
    print(training_args.num_train_epochs)

    assert int(training_args.num_epochs_trained_before_this_epoch) == int(training_args.num_train_epochs - 1), "only allow 1 new epochs"

    # Log the config
    from pprint import pprint
    if RANK == 0:
        print(f"Resuming from {model_args.model_name_or_path}")
        pprint(model_args.__dict__)
        pprint(data_args.__dict__)
        pprint(training_args.__dict__)

    config = {"rank": RANK}
    config.update(model_args.__dict__)
    config.update(data_args.__dict__)
    for key, value in training_args.__dict__.items():
        try:
            json.dumps(value)
        except Exception:
            print(
                f"Key '{key}' contains non-serializable value: {value} (type: {type(value)})"
            )
            continue
        config.update({key: value})

    wandb_path = training_args.log_dir.replace("log/", "log-wandb/", 1)
    if RANK == 0:
        print(f"wandb_path: {wandb_path}")
    os.makedirs(wandb_path, exist_ok=True)
    wandb.init(project="yulanmini",
               resume="allow",
               group=training_args.run_name,
               name=training_args.run_name,
               config=config,
               dir=wandb_path)
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }  # OR gradient_checkpointing_kwargs={'use_reentrant':True}, please refer to https://github.com/huggingface/transformers/issues/26969

    # Get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_args, data_args,
                                           training_args)

    if training_args.update_trained_steps_and_epochs:
        model.config.num_steps_trained_before_this_epoch  = training_args.num_steps_trained_before_this_epoch
        model.config.num_epochs_trained_before_this_epoch = training_args.num_epochs_trained_before_this_epoch
    model.config.gradient_checkpointing_step = model_args.gradient_checkpointing_step

    set_seed(training_args.seed)
    training_args.config = model.config
    dscfg = deepspeed_config()
    if RANK == 0:
        print(dscfg)

    # Modify `trainer_state.json`. This is necessary because HF Trainer, when there's a conflict between CLI arguments and parameters in the JSON file, prioritizes the JSON parameters. Therefore, we need to manually overwrite the JSON parameters with the CLI arguments.
    if training_args.resume_from_checkpoint:

        with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")) as f:
            state = json.load(f)

        torch.distributed.barrier()

        if state["train_batch_size"] != training_args.per_device_train_batch_size:
            if RANK == 0:
                print("Warning: train_batch_size is different from the checkpoint")
                state["train_batch_size"] = training_args.per_device_train_batch_size
                if not os.path.exists(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json.bak")):
                    shutil.copyfile(
                        os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"),
                        os.path.join(training_args.resume_from_checkpoint, "trainer_state.json.bak"),
                    )
                with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"), "w") as fn:
                    fn.write(json.dumps(state, indent=2))
            torch.distributed.barrier()

    # Prepare the data
    data_module = prepare_data(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
        skip=0,
    )

    # Update max_steps based on the length of the dataset
    num_update_steps_per_epoch = math.ceil((len(data_module["train_dataset"])/(WORLD_SIZE * training_args.per_device_train_batch_size)))// training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    print("num_update_steps_per_epoch: ")
    print(training_args.num_steps_trained_before_this_epoch)
    print(num_update_steps_per_epoch)
    print(len(data_module["train_dataset"]))
    print(WORLD_SIZE)
    print(training_args.per_device_train_batch_size)
    print(len(data_module["train_dataset"]) //(WORLD_SIZE * training_args.per_device_train_batch_size))
    print(training_args.gradient_accumulation_steps)
    training_args.max_steps = training_args.num_steps_trained_before_this_epoch + num_update_steps_per_epoch

    model.tokenizer = tokenizer

    if training_args.profile:
        callbacks = [PyTorchProfilerCallback, LogCallback]
    else:
        callbacks = [LogCallback]

    model.LOCAL_RANK = LOCAL_RANK
    model.log_dir = training_args.log_dir
    model.rank = RANK

    trainer = YuLanMiniTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

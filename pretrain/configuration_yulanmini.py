# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" YuLanMinimodel configuration"""

import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

YULANMINI_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class YuLanMiniConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YuLanMiniModel`]. It is used to instantiate an YuLanMini
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YuLanMini-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the YuLanMinimodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`YuLanMiniModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. YuLanMini1 supports up to 2048 tokens,
            YuLanMini2 up to 4096, CodeYuLanMiniup to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalYuLanMini/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    ```python
    >>> from transformers import YuLanMiniModel, YuLanMiniConfig
    >>> # Initializing a YuLanMini-7b style configuration
    >>> configuration = YuLanMiniConfig()
    >>> # Initializing a model from the YuLanMini-7b style configuration
    >>> model = YuLanMiniModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "yulanmini"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=99000,
        hidden_size=1920,
        intermediate_size=4800,
        num_hidden_layers=56,
        num_attention_heads=30,
        num_key_value_heads=6,
        # 不常用变量
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,  # /home/u20140041/pretrain-mini/preprocess/modify_tokenizer/1731
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        rope_scaling=None,
        attention_bias=True,  # qwen
        attention_dropout=0.0,
        # 放缩embedding grad
        shrink_alpha=1,
        shrink_alpha2=1,
        use_liger=False,
        # 初始化
        initializer_range=0.014434,
        init_scale_o=10.582218,
        model_reproduce="transformer",
        # 下面是为了muparam设置的参数，需要保证：默认值是不使用任何muparam的部分
        hidden_states_shrink=1,
        dim_model_base=None,
        dim_ffn_base_init=None,  # 新版muparam没有使用了
        dim_model_base_init=None,
        dim_model_base_attn=None,
        dim_model_base_lmh=None,
        dim_model_base_logits=None,
        dim_model_base_lr=None,
        scale_emb=1,
        # qk_layernorm
        qk_layernorm=False,
        layer_norm_eps=1e-6,
        embedding_ln=False,
        embedding_rmsln=False,
        ln_scale=1.,
        z_loss=0.0001,
        # wesar
        wesar_weights=True,
        embed_tokens_alpha=1,
        q_proj_alpha=1,
        k_proj_alpha=1,
        v_proj_alpha=1,
        o_proj_alpha=1,
        down_proj_alpha=1,
        gate_up_proj_alpha=1,
        input_layernorm_alpha=1,
        post_attention_layernorm_alpha=1,
        norm_alpha=1,
        lm_head_alpha=1,
        use_norm_alpha=True,
        use_emb_alpha=False,
        rms_type="llama",
        num_steps_trained_before_this_epoch=0,
        num_epochs_trained_before_this_epoch=0,
        # 加速
        gradient_checkpointing_step=7,
        **kwargs,
    ):
        # 训练states，每个epoch更新，epoch内部不会变。比如训练到第4轮数据，这两个的值都是第三轮最后一步的值（epochs=3, steps=xxx），只要是在第4轮，无论是多少步，都是第三轮的值，由update_trained_steps_and_epochs控制是否更新
        self.num_steps_trained_before_this_epoch = num_steps_trained_before_this_epoch
        self.num_epochs_trained_before_this_epoch = num_epochs_trained_before_this_epoch

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.shrink_alpha = shrink_alpha
        self.use_liger = use_liger
        self.init_scale_o = init_scale_o
        self.hidden_states_shrink = 1 / math.sqrt(num_hidden_layers) if hidden_states_shrink == "muparam" else hidden_states_shrink
        self.dim_model_base = dim_model_base if dim_model_base is not None else hidden_size
        self.dim_model_base_init = dim_model_base_init
        self.dim_model_base_attn = dim_model_base_attn if dim_model_base_attn is not None else (hidden_size // num_attention_heads)  # 初始化为1则是使用1/H_dim
        self.dim_model_base_lmh = dim_model_base_lmh if dim_model_base_lmh is not None else 1  # 初始化为1则是不放缩lm_head的init
        self.scale_emb = scale_emb if scale_emb is not None else 1
        self.model_reproduce=model_reproduce if model_reproduce is not None else "transformer"
        self.dim_model_base_logits = dim_model_base_logits if dim_model_base_logits is not None else hidden_size
        self.dim_model_base_lr = dim_model_base_lr if dim_model_base_lr is not None else hidden_size

        self.qk_layernorm = qk_layernorm
        self.layer_norm_eps = layer_norm_eps
        self.embedding_ln = embedding_ln
        self.embedding_rmsln = embedding_rmsln
        self.ln_scale = ln_scale
        self.z_loss = z_loss

        if embedding_ln and embedding_rmsln:
            raise ValueError("Only one of embedding_ln and embedding_rmsln should be True")

        self.wesar_weights = wesar_weights
        self.embed_tokens_alpha = embed_tokens_alpha
        self.q_proj_alpha = q_proj_alpha
        self.k_proj_alpha = k_proj_alpha
        self.v_proj_alpha = v_proj_alpha
        self.o_proj_alpha = o_proj_alpha
        self.down_proj_alpha = down_proj_alpha
        self.gate_up_proj_alpha = gate_up_proj_alpha
        self.input_layernorm_alpha = input_layernorm_alpha
        self.post_attention_layernorm_alpha = post_attention_layernorm_alpha
        self.norm_alpha = norm_alpha
        self.lm_head_alpha = lm_head_alpha
        self.use_norm_alpha = use_norm_alpha
        self.use_emb_alpha = use_emb_alpha
        self.rms_type = rms_type

        self.gradient_checkpointing_step = gradient_checkpointing_step

        if self.dim_model_base != hidden_size or self.dim_model_base_init is not None or self.dim_model_base_attn != (hidden_size // num_attention_heads) or self.dim_model_base_lmh != 1:
            if init_scale_o != 1:
                raise ValueError("When using muparam, init_scale_o should be 1")

        # multiplier
        print("Attention放缩：", math.sqrt(self.dim_model_base_attn) / (hidden_size // num_attention_heads))
        print("Residual链接处的Hidden States放缩：", hidden_states_shrink)
        print("Logits放缩：", 1 / (hidden_size / self.dim_model_base))

        # initializer
        if dim_model_base_init is not None:
            print("o_proj,down_proj初始化STD：", initializer_range / math.sqrt(2 * (hidden_size / dim_model_base_init) * num_hidden_layers))
            print("gate_proj,up_proj,q_proj,k_proj,v_proj初始化STD：", initializer_range / math.sqrt(self.hidden_size / self.dim_model_base_init))
        else:
            print("o_proj,down_proj初始化STD：", initializer_range / init_scale_o)
            print("gate_proj,up_proj,q_proj,k_proj,v_proj初始化STD：", initializer_range)
        print("lm_head初始化STD：", initializer_range / math.sqrt(self.dim_model_base_lmh))

        if not tie_word_embeddings and self.scale_emb != 1:
            raise ValueError("When using scale_emb, tie_word_embeddings should be False")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        try:
            import flash_attn
            self._attn_implementation = "flash_attention_2"
        except:
            pass

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

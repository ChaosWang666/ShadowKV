################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

""

import torch
import torch.nn.functional as F
import gc
import time

import transformers
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, sample_token
from .prompt_template import Templates, Chat_Templates
from .base import LLM

"""Qwen2模型的ShadowKV实现

本文件实现了Qwen2系列模型的高效推理功能，支持ShadowKV稀疏注意力优化。

主要特性：
1. 模型架构：
   - 独立的Q、K、V投影权重和偏置项（区别于Llama的融合QKV）
   - 支持分组查询注意力（GQA）
   - 高效的RoPE位置编码
   - SwiGLU激活函数的前馈网络

2. 优化策略：
   - ShadowKV稀疏注意力：通过块选择和SVD压缩减少内存使用
   - 高效的KV缓存管理：支持超长上下文推理
   - 内存优化：逐层加载权重，及时释放内存
   - GPU加速：所有计算在GPU上进行

3. 模板系统：
   - 内置Qwen专用的上下文和对话模板
   - 支持多轮对话和长文本处理

主要组件：
- Qwen2Layer: 单个Transformer层的权重容器
- Qwen2: 主要的模型推理类，继承自基础LLM类

适用场景：
- 长文本理解和生成
- 多轮对话系统
- 内存受限的推理环境
- 需要高吞吐量的应用
"""

class Qwen2Layer:
    """
    保存单层Qwen2 Transformer所需的权重参数
    
    该类封装了Qwen2模型单个Transformer层的所有权重参数，包括：
    - 自注意力机制的独立Q、K、V投影权重和偏置项
    - 注意力输出投影权重
    - 前馈网络的门控投影、上投影和下投影权重
    - 层归一化的权重和方差epsilon参数
    
    与Llama不同，Qwen2保持QKV投影的独立性，并包含偏置项。
    """

    def __init__(self, layer_idx) -> None:
        """
        初始化Qwen2层权重容器
        
        Args:
            layer_idx (int): 层索引，用于标识当前层在模型中的位置
        """
        # 自注意力机制权重（独立的Q、K、V投影）
        self.wq :torch.Tensor = None    # Query投影权重
        self.wk :torch.Tensor = None    # Key投影权重
        self.wv :torch.Tensor = None    # Value投影权重
        self.wo :torch.Tensor = None    # 注意力输出投影权重

        # QKV投影的偏置项（Qwen2特有）
        self.bq :torch.Tensor = None    # Query投影偏置
        self.bk :torch.Tensor = None    # Key投影偏置
        self.bv :torch.Tensor = None    # Value投影偏置

        # 前馈网络权重（独立的门控、上投影和下投影）
        self.gate_proj :torch.Tensor = None  # 门控投影权重
        self.up_proj :torch.Tensor = None    # 上投影权重
        self.down_proj :torch.Tensor = None  # 下投影权重

        # 输入层归一化参数
        self.input_layernorm_weight :torch.Tensor = None           # 输入层归一化权重
        self.input_layernorm_variance_epsilon :float = 0.0         # 输入层归一化方差epsilon

        # 后注意力层归一化参数
        self.post_attention_layernorm_weight :torch.Tensor = None  # 后注意力层归一化权重
        self.post_attention_layernorm_variance_epsilon :float = 0.0 # 后注意力层归一化方差epsilon

        self.layer_idx = layer_idx  # 层索引

    def init_parameters(self, hf_layer: Qwen2DecoderLayer):
        """
        从HuggingFace的Qwen2DecoderLayer初始化权重参数
        
        提取HuggingFace格式的权重，包括：
        - 独立的Q、K、V投影权重和偏置项
        - 注意力输出投影权重
        - 前馈网络的独立投影权重
        - 层归一化参数
        
        Args:
            hf_layer (Qwen2DecoderLayer): HuggingFace的Qwen2解码器层
        """

        # 提取独立的QKV投影权重
        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()  # Query投影权重
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()  # Key投影权重
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()  # Value投影权重
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()  # 输出投影权重

        # 提取QKV投影的偏置项（Qwen2特有）
        self.bq = hf_layer.self_attn.q_proj.bias.detach()  # Query投影偏置
        self.bk = hf_layer.self_attn.k_proj.bias.detach()  # Key投影偏置
        self.bv = hf_layer.self_attn.v_proj.bias.detach()  # Value投影偏置

        # 提取前馈网络权重（保持独立）
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()  # 门控投影权重
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()      # 上投影权重
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()  # 下投影权重

        # 提取层归一化参数
        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):
        """
        将所有权重参数转移到指定的GPU设备
        
        Args:
            device (str): 目标设备，默认为'cuda:0'
        """

        # 使用非阻塞传输提高性能
        # 层归一化权重
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        
        # 注意力权重
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        
        # 前馈网络权重
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

        # QKV偏置项
        self.bq = self.bq.to(device, non_blocking=True)
        self.bk = self.bk.to(device, non_blocking=True)
        self.bv = self.bv.to(device, non_blocking=True)

class Qwen2(LLM):
    """
    Qwen2模型的推理实现
    
    继承自基础LLM类，实现了Qwen2系列模型的高效推理功能。
    支持多种优化策略：
    - 标准全注意力模式
    - ShadowKV稀疏注意力模式
    - 高效的KV缓存管理
    
    主要特性：
    - 支持超长上下文（最大64K tokens）
    - 独立的QKV投影权重和偏置项
    - 内存优化的KV缓存管理
    - 高效的RoPE位置编码
    - 内置Qwen对话模板
    """

    def __init__(self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        batch_size :int = 1,
        max_length :int = 64*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        sparse_budget: int = 2048,
        rank=160,
        chunk_size=8,
        minference=False) -> None:
        """
        初始化Qwen2模型
        
        Args:
            model_name (str): HuggingFace模型名称，默认为"Qwen/Qwen2-7B-Instruct"
            batch_size (int): 批处理大小，当前仅支持1
            max_length (int): 最大序列长度，默认64K tokens
            device (str): 计算设备，默认为'cuda:0'
            dtype: 数据类型，默认为torch.bfloat16
            attn_mode (str): 注意力模式，'full'为全注意力，'sparse'为稀疏注意力
            sparse_budget (int): 稀疏注意力预算，控制保留的token数量
            rank (int): SVD分解的秩，用于压缩key状态
            chunk_size (int): 块大小，用于分块处理
            minference (bool): 是否启用MinInference优化
        """
        
        assert batch_size == 1, "Batch size must be 1"  # 当前仅支持批大小为1
        
        # 基础配置
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen2Config.from_pretrained(model_name)  # 加载模型配置
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)  # 初始化分词器
        
        # 模型架构参数
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size                    # 隐藏层维度
        self.num_heads = self.config.num_attention_heads              # 注意力头数
        self.head_dim = self.hidden_size // self.num_heads            # 每个注意力头的维度
        self.num_key_value_heads = self.config.num_key_value_heads    # KV头数（用于GQA）
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # KV组数
        self.max_position_embeddings = self.config.max_position_embeddings      # 最大位置编码长度
        self.rope_theta = self.config.rope_theta                      # RoPE的theta参数

        # 初始化模型参数
        self.init_parameters()
        
        # 注意力模式和优化设置
        self.attn_mode = attn_mode
        self.minference = minference

        # 设置Qwen专用模板
        self.ctx_template = Templates['qwen']        # 上下文模板
        self.chat_template = Chat_Templates['qwen']  # 对话模板

        # 初始化KV缓存
        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config)

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        """
        预计算RoPE的cos和sin缓存
        
        为了提高推理效率，预先计算所有位置的cos和sin值。
        
        Args:
            inv_freq (torch.Tensor): RoPE的逆频率张量
            
        Returns:
            tuple: (cos_cache, sin_cache) 预计算的cos和sin缓存
        """
        # 生成位置索引
        t = torch.arange(self.max_length, device=self.device, dtype=torch.int64).type_as(inv_freq)
        # 计算频率矩阵
        freqs = torch.outer(t, inv_freq)
        # 拼接频率以匹配RoPE的维度要求
        emb = torch.cat((freqs, freqs), dim=-1)
        # 返回cos和sin缓存
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):
        """
        从HuggingFace模型加载权重参数
        
        加载预训练模型的所有权重，包括：
        - 词嵌入层权重
        - 语言模型头权重
        - 最终层归一化权重
        - 所有Transformer层的权重
        - RoPE的cos/sin缓存
        """
        # 加载HuggingFace预训练模型
        hf_model = Qwen2ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        
        # 提取并转移核心权重到GPU
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)  # 词嵌入权重
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)                  # 语言模型头权重
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)          # 最终层归一化权重
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon               # 层归一化epsilon
        
        # 预计算RoPE缓存
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(
            hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.to(self.device)
        )
        
        # 初始化Transformer层列表
        self.layers :list[Qwen2Layer] = []

        # 逐层加载权重并转移到GPU
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen2Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)  # 从HF层提取权重
            layer.init_gpu(self.device)               # 转移到GPU
            self.layers.append(layer)
            hf_model.model.layers[idx] = None          # 释放原始层内存
            gc.collect()                               # 强制垃圾回收

        self.num_layers = len(self.layers)  # 记录层数

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen2Layer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        """
        注意力计算前的预处理
        
        执行注意力机制前的准备工作：
        1. 输入层归一化
        2. QKV投影（包含偏置项）
        3. 重塑张量形状以适配多头注意力
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态 [batch_size, seq_len, hidden_size]
            buffer (Qwen2Layer): 当前层的权重缓冲区
            num_heads (int): 注意力头数
            num_key_value_heads (int): KV头数
            head_dim (int): 每个注意力头的维度
            
        Returns:
            tuple: (query_states, key_states, value_states) 处理后的QKV状态
        """
        # 输入层归一化
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        
        bsz, q_len, _ = hidden_states.size()
        
        # QKV投影（包含偏置项，这是Qwen2的特色）
        query_states = F.linear(hidden_states, buffer.wq, bias=buffer.bq)  # Query投影
        key_states = F.linear(hidden_states, buffer.wk, bias=buffer.bk)    # Key投影
        value_states = F.linear(hidden_states, buffer.wv, bias=buffer.bv)  # Value投影
        
        # 重塑为多头注意力格式 [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        return query_states, key_states, value_states
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen2Layer
    ):  
        """
        注意力计算后的处理
        
        执行注意力机制后的后处理工作：
        1. 注意力输出投影
        2. 第一个残差连接
        3. 后注意力层归一化
        4. SwiGLU前馈网络
        5. 第二个残差连接
        
        Args:
            attn_output (torch.Tensor): 注意力机制的输出
            residual (torch.Tensor): 残差连接的输入
            buffer (Qwen2Layer): 当前层的权重缓冲区
            
        Returns:
            torch.Tensor: 处理后的隐藏状态
        """
        # 注意力输出投影
        hidden_states = F.linear(attn_output, buffer.wo)
        # 第一个残差连接（注意力分支）
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # 后注意力层归一化
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        
        # SwiGLU前馈网络
        up = F.linear(hidden_states, buffer.up_proj)        # 上投影
        gate = F.silu(F.linear(hidden_states, buffer.gate_proj))  # 门控投影 + SiLU激活
        hidden_states = gate * up                           # 门控机制
        hidden_states = F.linear(hidden_states, buffer.down_proj)  # 下投影
        
        # 第二个残差连接（前馈分支）
        hidden_states = residual + hidden_states
        return hidden_states
    
    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对单个张量应用旋转位置编码（RoPE）
        
        Args:
            x (torch.Tensor): 输入张量（通常是query或key）
            position_ids (torch.Tensor): 位置索引
            
        Returns:
            torch.Tensor: 应用RoPE后的张量
        """
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对query和key张量同时应用旋转位置编码（RoPE）
        
        RoPE通过旋转变换为序列中的每个位置编码相对位置信息，
        使模型能够理解token之间的相对位置关系。
        
        Args:
            q (torch.Tensor): Query张量 [batch_size, num_heads, seq_len, head_dim]
            k (torch.Tensor): Key张量 [batch_size, num_kv_heads, seq_len, head_dim]
            position_ids (torch.Tensor): 位置索引 [batch_size, seq_len]
            
        Returns:
            tuple: (rotated_q, rotated_k) 应用RoPE后的query和key张量
        """
        return apply_rotary_pos_emb(q, k, self.cos_cache, self.sin_cache, position_ids)

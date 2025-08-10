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

"""
Llama模型的ShadowKV实现

本模块为Llama模型系列（包括Llama-2、Llama-3、Yi等）提供了优化的实现，支持：
- 标准全注意力机制
- ShadowKV稀疏注意力机制
- MinInference优化
- 高效的KV缓存管理
- 超长上下文支持（最大1M tokens）

主要组件：
- LlamaLayer: 存储单个Transformer层的权重参数
- Llama: 主要的模型类，提供推理功能

该实现专注于内存效率和推理速度，适用于长上下文应用场景。

特性说明：
1. 权重优化：将QKV投影权重合并，提高计算效率
2. 内存管理：支持CPU-GPU混合存储，降低显存占用
3. 位置编码：使用RoPE（旋转位置编码）支持超长序列
4. 模板系统：内置多种对话模板，支持不同的模型变体
5. 批量推理：支持单样本和批量推理模式
6. 稀疏注意力：通过ShadowKV实现内存高效的长上下文处理
"""

import torch
import torch.nn.functional as F
import gc

import transformers
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
transformers.logging.set_verbosity_error()

import vllm
from minference.configs.model2path import MODEL2PATH

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, apply_rotary_pos_emb_cuda
from .prompt_template import Templates, Chat_Templates, Prefix_Templates
from .base import LLM

class LlamaLayer:
    """
    保存单层Llama Transformer所需的权重参数
    
    该类封装了Llama模型单个Transformer层的所有权重参数，包括：
    - 自注意力机制的QKV投影权重和输出投影权重
    - 前馈网络的门控投影、上投影和下投影权重
    - 层归一化的权重和方差epsilon参数
    
    通过将权重参数分离出来，可以更灵活地管理内存和计算。
    """

    def __init__(self, layer_idx) -> None:
        """
        初始化Llama层权重容器
        
        Args:
            layer_idx (int): 层索引，用于标识当前层在模型中的位置
        """
        # 自注意力机制权重
        self.wqkv :torch.Tensor = None    # 合并的QKV投影权重矩阵
        self.wo :torch.Tensor = None      # 注意力输出投影权重

        # 前馈网络权重
        self.gate_up_proj :torch.Tensor = None  # 合并的门控和上投影权重
        self.down_proj :torch.Tensor = None     # 下投影权重

        # 输入层归一化参数
        self.input_layernorm_weight :torch.Tensor = None           # 输入层归一化权重
        self.input_layernorm_variance_epsilon :float = 0.0         # 输入层归一化方差epsilon

        # 后注意力层归一化参数
        self.post_attention_layernorm_weight :torch.Tensor = None  # 后注意力层归一化权重
        self.post_attention_layernorm_variance_epsilon :float = 0.0 # 后注意力层归一化方差epsilon

        self.layer_idx = layer_idx  # 层索引

    def init_parameters(self, hf_layer: LlamaDecoderLayer):
        """
        从HuggingFace的LlamaDecoderLayer初始化权重参数
        
        将HuggingFace格式的权重转换为优化后的格式，包括：
        - 将Q、K、V投影权重合并为单个矩阵以提高计算效率
        - 将门控投影和上投影权重合并
        - 提取层归一化参数
        
        Args:
            hf_layer (LlamaDecoderLayer): HuggingFace的Llama解码器层
        """

        # 合并QKV投影权重以提高计算效率
        # 将Q、K、V三个独立的投影矩阵合并为一个大矩阵
        self.wqkv :torch.Tensor= torch.cat((
            hf_layer.self_attn.q_proj.weight.detach(), 
            hf_layer.self_attn.k_proj.weight.detach(), 
            hf_layer.self_attn.v_proj.weight.detach()
        ), dim=0)
        
        # 注意力输出投影权重
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        
        # 记录Q和KV的维度大小，用于后续的权重分割
        self.q_size = hf_layer.self_attn.q_proj.weight.shape[0]
        self.kv_size = hf_layer.self_attn.k_proj.weight.shape[0]

        # 合并前馈网络的门控投影和上投影权重
        self.gate_up_proj = torch.cat((
            hf_layer.mlp.gate_proj.weight.detach(), 
            hf_layer.mlp.up_proj.weight.detach()
        ), dim=0)
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

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
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wqkv = self.wqkv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_up_proj = self.gate_up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

class Llama(LLM):
    """
    Llama模型的推理实现
    
    继承自基础LLM类，实现了Llama系列模型（包括Llama-2、Llama-3、Yi等）
    的高效推理功能。支持多种优化策略：
    - 标准全注意力模式
    - ShadowKV稀疏注意力模式
    - MinInference优化模式
    - 批量推理和单样本推理
    
    主要特性：
    - 支持超长上下文（最大1M tokens）
    - 内存优化的KV缓存管理
    - 高效的RoPE位置编码
    - 灵活的模板系统
    """

    def __init__(self,
        model_name: str = "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
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
        初始化Llama模型
        
        Args:
            model_name (str): 模型名称或路径，支持Llama-2、Llama-3、Yi等模型
            batch_size (int): 批次大小，默认为1
            max_length (int): 最大序列长度，默认64K
            device (str): 计算设备，默认'cuda:0'
            dtype: 数据类型，默认torch.bfloat16
            attn_mode (str): 注意力模式，'full'或'shadowkv'
            sparse_budget (int): ShadowKV模式下的稀疏预算
            rank (int): SVD分解的秩
            chunk_size (int): chunk大小
            minference (bool): 是否启用MinInference优化
        """
        
        # 基础配置初始化
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        # 分词器初始化
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        
        # 模型架构参数
        self.max_length = max_length                                    # 最大序列长度
        self.hidden_size = self.config.hidden_size                     # 隐藏层维度
        self.num_heads = self.config.num_attention_heads               # 注意力头数
        self.head_dim = self.hidden_size // self.num_heads             # 每个注意力头的维度
        self.num_key_value_heads = self.config.num_key_value_heads     # KV头数（用于GQA）
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 每个KV头对应的Q头数
        self.max_position_embeddings = self.config.max_position_embeddings      # 最大位置编码长度
        self.rope_theta = self.config.rope_theta                       # RoPE的theta参数
        self.vocab_size = self.config.vocab_size                       # 词汇表大小

        # 初始化模型参数
        self.init_parameters()
        
        # 注意力模式和优化设置
        self.attn_mode = attn_mode      # 注意力模式：'full'或'shadowkv'
        self.minference = minference    # 是否启用MinInference优化

        # 根据模型类型设置对话模板
        if 'llama-3' in model_name.lower():
            self.ctx_template = Templates['llama-3']           # 上下文模板
            self.chat_template = Chat_Templates['llama-3']     # 对话模板
            self.prefix_template = Prefix_Templates['llama-3'] # 前缀模板
        elif 'yi' in model_name.lower():
            self.ctx_template = Templates['yi']           # Yi模型的上下文模板
            self.chat_template = Chat_Templates['yi']     # Yi模型的对话模板
            self.prefix_template = Prefix_Templates['yi'] # Yi模型的前缀模板
        else:
            raise ValueError(f"Invalid model name {model_name}")

        # 初始化KV缓存
        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config)

        # 如果启用MinInference优化，加载预计算的注意力模式
        if self.minference:
            import json
            self.minference_parttern = []
            # 为每一层加载MinInference的注意力模式配置
            for layer_idx in range(self.num_layers):
                self.minference_parttern.append({int(ii): jj for ii, jj in json.load(open(MODEL2PATH[self.model_name]))[layer_idx].items()})


    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        """
        设置RoPE（旋转位置编码）的cos和sin缓存
        
        Args:
            inv_freq (torch.Tensor): 逆频率张量
            
        Returns:
            tuple: (cos_cache, sin_cache) 余弦和正弦缓存
        """
        # 生成位置索引，额外增加1024以应对超长序列
        t = torch.arange(self.max_length + 1024, device=self.device, dtype=inv_freq.dtype)
        # 计算频率矩阵
        freqs = torch.outer(t, inv_freq)
        # 拼接频率以匹配RoPE的维度要求
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对单个张量应用旋转位置编码
        
        Args:
            x (torch.Tensor): 输入张量
            position_ids (torch.Tensor): 位置ID
            
        Returns:
            torch.Tensor: 应用RoPE后的张量
        """
        return apply_rotary_pos_emb_cuda(x, self.cos_sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对Query和Key张量应用旋转位置编码
        
        Args:
            q (torch.Tensor): Query张量
            k (torch.Tensor): Key张量
            position_ids (torch.Tensor): 位置ID
            
        Returns:
            tuple: (q, k) 应用RoPE后的Query和Key张量
        """
        # 使用vLLM的优化RoPE实现
        vllm._custom_ops.rotary_embedding(position_ids, q, k, 128, self.cos_sin_cache, True)
        bsz = q.shape[0]
        # 重塑张量形状以匹配多头注意力的要求
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k

    def init_parameters(self):
        """
        从HuggingFace模型初始化所有参数
        
        加载预训练模型的权重，包括：
        - 词嵌入层权重
        - 语言模型头权重
        - 最终层归一化权重
        - 所有Transformer层的权重
        - RoPE位置编码缓存
        """
        # 加载HuggingFace预训练模型
        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        
        # 提取并转移核心权重到GPU
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)  # 词嵌入权重
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)                  # 语言模型头权重
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)          # 最终层归一化权重
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon               # 层归一化epsilon
        # 初始化RoPE位置编码缓存
        try:
            # 尝试从预训练模型中获取已缓存的cos和sin值
            cos_cache = hf_model.model.layers[0].self_attn.rotary_emb.cos_cached[:self.max_length+1024].to(self.device).to(self.dtype)
            sin_cache = hf_model.model.layers[0].self_attn.rotary_emb.sin_cached[:self.max_length+1024].to(self.device).to(self.dtype)
        except:
            # 如果缓存不存在，则重新计算
            cos_cache, sin_cache = self._set_cos_sin_cache(hf_model.model.layers[0].self_attn.rotary_emb.inv_freq.to(self.device))
        
        # 合并cos和sin缓存，只保留前64维（head_dim的一半）
        self.cos_sin_cache = torch.cat((cos_cache[:, :64], sin_cache[:, :64]), dim=-1)
        
        # 释放临时缓存内存
        del cos_cache, sin_cache

        # 初始化所有Transformer层
        self.layers :list[LlamaLayer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            # 创建自定义层对象
            layer = LlamaLayer(idx)
            # 从HuggingFace层初始化参数
            layer.init_parameters(hf_layer=hf_layer)
            # 将参数转移到GPU
            layer.init_gpu(self.device)
            self.layers.append(layer)
            # 释放原始层的内存
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: LlamaLayer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):
        """
        注意力计算前的预处理
        
        执行输入层归一化和QKV投影变换
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态
            buffer (LlamaLayer): 当前层的权重缓冲区
            num_heads (int): 注意力头数
            num_key_value_heads (int): KV头数
            head_dim (int): 每个头的维度
            
        Returns:
            tuple: (query_states, key_states, value_states) Q、K、V状态
        """
        # 输入层归一化
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        # QKV线性投影
        qkv = F.linear(hidden_states, buffer.wqkv)
        # 分割QKV
        query_states, key_states, value_states = qkv.split([buffer.q_size, buffer.kv_size, buffer.kv_size], dim=-1)

        # 重塑value_states为多头格式
        return query_states, key_states, value_states.view(value_states.shape[0], -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: LlamaLayer
    ):
        """
        注意力计算后的后处理
        
        执行注意力输出投影、残差连接、层归一化和前馈网络计算
        
        Args:
            attn_output (torch.Tensor): 注意力输出
            residual (torch.Tensor): 残差连接的输入
            buffer (LlamaLayer): 当前层的权重缓冲区
            
        Returns:
            torch.Tensor: 处理后的隐藏状态
        """
        # 注意力输出投影
        hidden_states = F.linear(attn_output, buffer.wo)
        # 第一个残差连接
        hidden_states = residual + hidden_states
        residual = hidden_states
        # 后注意力层归一化
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        
        # 前馈网络：门控和上投影
        hidden_states = F.linear(hidden_states, buffer.gate_up_proj)
        # 分割门控投影和上投影的结果
        d = hidden_states.shape[-1] // 2
        output_shape = (hidden_states.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        # 使用vLLM优化的SiLU激活函数和元素乘法
        vllm._custom_ops.silu_and_mul(out, hidden_states)
        
        # 下投影
        hidden_states = F.linear(out, buffer.down_proj)
        # 第二个残差连接
        hidden_states = residual + hidden_states
        return hidden_states



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

"""GLM 模型的简化推理实现

本文件实现了GLM（General Language Model）系列模型的ShadowKV优化推理。
GLM是智谱AI开发的大语言模型，具有以下特点：

架构特性：
- 融合QKV投影：将Query、Key、Value投影合并为单个线性层，提高计算效率
- 多查询注意力（MQA）：Key和Value头数少于Query头数，减少内存占用
- RMSNorm层归一化：使用RMSNorm替代LayerNorm，提高数值稳定性
- GLU前馈网络：使用门控线性单元，增强模型表达能力
- RoPE位置编码：旋转位置编码，支持长序列建模

优化策略：
- ShadowKV稀疏注意力：智能选择重要token，大幅减少计算量
- 高效KV缓存管理：优化内存使用，支持长上下文推理
- 内存优化：精确控制GPU内存分配，避免OOM
- GPU加速：充分利用GPU并行计算能力
- 模板系统：内置GLM专用的对话和上下文模板

特性说明：
- 权重优化：采用非阻塞GPU传输，提高加载速度
- 内存管理：逐层加载权重并及时释放，控制峰值内存
- RoPE缓存：预计算旋转位置编码，避免重复计算
- 模板系统：支持GLM特有的对话格式和上下文处理
- 批量推理：虽然当前限制batch_size=1，但架构支持未来扩展
- 稀疏注意力：通过ShadowKV技术实现高效的长序列处理
"""

import torch
import torch.nn.functional as F
import gc
import time

import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

transformers.logging.set_verbosity_error()

import vllm

from .tensor_op import layer_norm
from .prompt_template import Templates, Chat_Templates
from .base import LLM

class GLMLayer:
    """
    存储单层GLM Transformer所需的权重
    
    GLMLayer封装了GLM模型单个Transformer层的所有权重参数，包括：
    - 融合QKV投影权重和偏置（GLM特有的设计）
    - 注意力输出投影权重
    - 前馈网络的上投影和下投影权重
    - 输入和后注意力层归一化参数
    
    与其他模型的主要区别：
    - 使用融合的QKV投影，包含偏置项
    - 采用多查询注意力（MQA）架构
    - 使用RMSNorm进行层归一化
    """

    def __init__(self, layer_idx) -> None:
        """
        初始化GLM层权重容器
        
        Args:
            layer_idx (int): 层索引，用于标识当前层
        """
        
        # 注意力机制权重
        self.wqkv :torch.Tensor = None    # 融合的QKV投影权重
        self.bqkv :torch.Tensor = None    # 融合的QKV投影偏置（GLM特有）
        self.wo :torch.Tensor = None      # 注意力输出投影权重

        # 前馈网络权重
        self.up_proj :torch.Tensor = None    # 上投影权重（扩展维度）
        self.down_proj :torch.Tensor = None  # 下投影权重（恢复维度）

        # 层归一化参数
        self.input_layernorm_weight :torch.Tensor = None           # 输入层归一化权重
        self.input_layernorm_variance_epsilon :float = 0.0         # 输入层归一化epsilon

        self.post_attention_layernorm_weight :torch.Tensor = None  # 后注意力层归一化权重
        self.post_attention_layernorm_variance_epsilon :float = 0.0 # 后注意力层归一化epsilon

        self.layer_idx = layer_idx  # 层索引

    def init_parameters(self, hf_layer: LlamaDecoderLayer):
        """
        从HuggingFace的GLM层提取权重参数
        
        将HuggingFace格式的GLM层权重转换为ShadowKV格式。
        GLM使用融合的QKV投影和特殊的MLP结构。
        
        Args:
            hf_layer: HuggingFace的GLM解码器层
        """

        # 提取注意力机制权重
        self.wqkv: torch.Tensor = hf_layer.self_attention.query_key_value.weight.detach()  # 融合QKV权重
        self.bqkv: torch.Tensor = hf_layer.self_attention.query_key_value.bias.detach()    # 融合QKV偏置
        self.wo :torch.Tensor= hf_layer.self_attention.dense.weight.detach()              # 输出投影权重

        # 提取前馈网络权重
        self.up_proj = hf_layer.mlp.dense_h_to_4h.weight.detach()    # 上投影（hidden -> 4*hidden）
        self.down_proj = hf_layer.mlp.dense_4h_to_h.weight.detach()  # 下投影（4*hidden -> hidden）

        # 提取层归一化参数
        self.input_layernorm_weight = hf_layer.input_layernorm.weight                      # 输入层归一化权重
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.eps              # 输入层归一化epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight    # 后注意力层归一化权重
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.eps  # 后注意力层归一化epsilon
    
    def init_gpu(self, device:str = 'cuda:0'):
        """
        将权重转移到GPU设备
        
        使用非阻塞传输将所有权重从CPU转移到指定的GPU设备，
        提高数据传输效率。
        
        Args:
            device (str): 目标GPU设备，默认为'cuda:0'
        """

        # 转移层归一化权重
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        
        # 转移注意力权重
        self.wqkv = self.wqkv.to(device, non_blocking=True)  # 融合QKV权重
        self.bqkv = self.bqkv.to(device, non_blocking=True)  # 融合QKV偏置
        self.wo = self.wo.to(device, non_blocking=True)      # 输出投影权重
        
        # 转移前馈网络权重
        self.up_proj = self.up_proj.to(device, non_blocking=True)    # 上投影权重
        self.down_proj =  self.down_proj.to(device, non_blocking=True)  # 下投影权重

class GLMConfig:
    """
    GLM模型配置类
    
    封装GLM模型的关键配置参数，特别是多查询注意力（MQA）相关的配置。
    GLM使用MQA架构，其中Key和Value头数少于Query头数，以减少内存占用。
    """
    
    def __init__(self, config) -> None:
        """
        从HuggingFace配置初始化GLM配置
        
        Args:
            config: HuggingFace的GLM模型配置
        """
        self.hidden_size = config.hidden_size                                    # 隐藏层维度
        self.num_heads = config.num_attention_heads                             # Query头数
        self.head_dim = self.hidden_size // self.num_heads                      # 每个注意力头的维度
        self.num_key_value_heads = config.multi_query_group_num                 # KV头数（MQA特性）
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # KV组数
        self.num_hidden_layers = config.num_hidden_layers                       # Transformer层数
        self.num_attention_heads = config.num_attention_heads                   # 注意力头数（与num_heads相同）

class GLM(LLM):
    """
    GLM模型主推理类
    
    继承自LLM基类，实现GLM系列模型的高效推理。
    支持ShadowKV稀疏注意力、长上下文处理、内存优化等功能。
    
    主要特性：
    - 融合QKV投影和多查询注意力（MQA）
    - ShadowKV稀疏注意力优化
    - 高效的KV缓存管理
    - 支持超长上下文（默认64K tokens）
    - 内置GLM专用模板系统
    - GPU内存优化和加速
    """

    def __init__(self,
        model_name: str = "THUDM/glm-4-9b-chat-1m",
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
        初始化GLM模型
        
        Args:
            model_name (str): HuggingFace模型名称，默认为"THUDM/glm-4-9b-chat-1m"
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
        
        # 基础配置
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        # 加载HuggingFace模型和配置
        hf_model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype, trust_remote_code=True)
        self.config = hf_model.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False, trust_remote_code=True)
        
        # 模型架构参数
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size                    # 隐藏层维度
        self.num_heads = self.config.num_attention_heads              # Query头数
        self.head_dim = self.hidden_size // self.num_heads            # 每个注意力头的维度
        self.num_key_value_heads = self.config.multi_query_group_num  # KV头数（MQA特性）
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # KV组数
        self.max_position_embeddings = self.config.seq_length         # 最大位置编码长度
        self.rope_ratio = self.config.rope_ratio                      # RoPE比例参数

        # 初始化模型参数
        self.init_parameters(hf_model)
        
        # 注意力模式和优化设置
        self.attn_mode = attn_mode
        self.minference = minference

        # 设置GLM专用模板
        self.ctx_template = Templates['glm']        # 上下文模板
        self.chat_template = Chat_Templates['glm']  # 对话模板
        self.prefix_template = Templates['glm']     # 前缀模板

        # 词汇表大小
        self.vocab_size = self.config.vocab_size

        # 初始化KV缓存
        self.init_kv_cache(sparse_budget, rank, chunk_size, GLMConfig(self.config))

    def _set_cos_sin_cache(self, hf_model):
        """
        预计算RoPE的cos和sin缓存
        
        GLM使用特殊的RoPE实现，需要预计算旋转位置编码的缓存。
        
        Args:
            hf_model: HuggingFace的GLM模型
            
        Returns:
            torch.Tensor: 预计算的RoPE缓存，形状为[max_length, 64]
        """
        return hf_model.transformer.rotary_pos_emb(self.max_length + 1024).to(self.device).transpose(-1, -2).contiguous().view(-1, 64)

    def init_parameters(self, hf_model):
        """
        从HuggingFace模型加载权重参数
        
        加载预训练模型的所有权重，包括：
        - 词嵌入层权重
        - 语言模型头权重
        - 最终层归一化权重
        - 所有Transformer层的权重
        - RoPE的cos/sin缓存
        
        Args:
            hf_model: HuggingFace的GLM模型
        """
        # 提取并转移核心权重到GPU
        self.embed_tokens = hf_model.transformer.embedding.word_embeddings.weight.detach().to(self.device)  # 词嵌入权重
        self.lm_head = hf_model.transformer.output_layer.weight.detach().to(self.device)                    # 语言模型头权重
        self.norm_weight = hf_model.transformer.encoder.final_layernorm.weight.detach().to(self.device)     # 最终层归一化权重
        self.norm_variance_epsilon = hf_model.transformer.encoder.final_layernorm.eps                       # 层归一化epsilon
        
        # 预计算RoPE缓存
        self.cos_sin_cache = self._set_cos_sin_cache(hf_model)

        # 初始化Transformer层列表
        self.layers :list[GLMLayer] = []

        # 逐层加载权重并转移到GPU
        for idx, hf_layer in enumerate(hf_model.transformer.encoder.layers):
            layer = GLMLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)  # 从HF层提取权重
            layer.init_gpu(self.device)               # 转移到GPU
            self.layers.append(layer)
            hf_model.transformer.encoder.layers[idx] = None  # 释放原始层内存
            gc.collect()                               # 强制垃圾回收

        self.num_layers = len(self.layers)  # 记录层数

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: GLMLayer,
        num_heads: int,
        num_key_value_heads:int,
        head_dim:int
    ):  
        """
        注意力计算前的预处理
        
        执行注意力机制前的准备工作：
        1. 输入层归一化
        2. 融合QKV投影（包含偏置）
        3. 分离Q、K、V状态
        4. 重塑Value张量形状以适配多头注意力
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态 [batch_size, seq_len, hidden_size]
            buffer (GLMLayer): 当前层的权重缓冲区
            num_heads (int): Query头数
            num_key_value_heads (int): KV头数
            head_dim (int): 每个注意力头的维度
            
        Returns:
            tuple: (query_states, key_states, value_states) 处理后的QKV状态
        """
        # 输入层归一化
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        
        # 融合QKV投影（包含偏置，GLM特有）
        mixed_x_layer = F.linear(input=hidden_states, weight=buffer.wqkv, bias=buffer.bqkv)
        
        # 分离Q、K、V状态（按维度切分融合的QKV输出）
        (query_states, key_states, value_states) = mixed_x_layer.split(
                [
                    num_heads * head_dim,           # Query部分
                    num_key_value_heads * head_dim, # Key部分
                    num_key_value_heads * head_dim, # Value部分
                ],
                dim=-1,
            )

        # 重塑Value张量为多头注意力格式 [batch_size, num_kv_heads, seq_len, head_dim]
        return query_states, key_states, value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: GLMLayer
    ):  
        """
        注意力计算后的处理
        
        执行注意力机制后的后处理工作：
        1. 注意力输出投影
        2. 第一个残差连接
        3. 后注意力层归一化
        4. GLU前馈网络（使用SiLU激活和门控机制）
        5. 第二个残差连接
        
        Args:
            attn_output (torch.Tensor): 注意力机制的输出
            residual (torch.Tensor): 残差连接的输入
            buffer (GLMLayer): 当前层的权重缓冲区
            
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

        # GLU前馈网络
        hidden_states = F.linear(hidden_states, buffer.up_proj)  # 上投影
        d = hidden_states.shape[-1] // 2                         # 分割维度
        output_shape = (hidden_states.shape[:-1] + (d, ))        # 输出形状
        out = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vllm._custom_ops.silu_and_mul(out, hidden_states)        # SiLU激活和门控乘法
        
        # 下投影
        hidden_states = F.linear(out, buffer.down_proj)
        # 第二个残差连接（前馈分支）
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对query和key张量同时应用旋转位置编码（RoPE）
        
        GLM使用vLLM的优化RoPE实现，通过自定义CUDA算子提高计算效率。
        
        Args:
            q (torch.Tensor): Query张量
            k (torch.Tensor): Key张量
            position_ids (torch.Tensor): 位置索引
            
        Returns:
            tuple: (rotated_q, rotated_k) 应用RoPE后的query和key张量
        """
        # 使用vLLM的优化RoPE实现
        vllm._custom_ops.rotary_embedding(position_ids, q, k, 128, self.cos_sin_cache, False)
        
        # 重塑为多头注意力格式
        bsz = q.shape[0]
        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)           # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [batch_size, num_kv_heads, seq_len, head_dim]
        return q, k

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        对单个张量应用旋转位置编码（RoPE）
        
        GLM的RoPE实现，支持灵活的张量形状和位置索引格式。
        
        Args:
            x (torch.Tensor): 输入张量（通常是query或key）
            position_ids (torch.Tensor): 位置索引
            
        Returns:
            torch.Tensor: 应用RoPE后的张量
        """
        # 处理输入张量形状
        if len(x.shape) == 3: # x: [bsz, seq, 1024]
            x = x.view(x.size(0), x.size(1), -1, 128).transpose(1, 2) # [bsz, heads, seq, 128]
        
        # 处理位置索引的不同格式
        if len(position_ids.shape) == 1: # position_ids: [seq]
            position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
        if len(position_ids.shape) == 2: # position_ids: [bsz, seq]
            position_ids = position_ids.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 获取RoPE缓存
        rope_cache = self.cos_sin_cache[position_ids] # [max_len, 64] --> [bsz, heads, seq, 64]
        rot_dim = 64
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]  # 分离旋转部分和非旋转部分

        # 应用旋转变换
        x_out2 = torch.stack(
            [
                x[..., 0::2] * rope_cache[..., :32] - x[..., 1::2] * rope_cache[..., 32:],  # 实部
                x[..., 1::2] * rope_cache[..., :32] + x[..., 0::2] * rope_cache[..., 32:],  # 虚部
            ],
            -1,
        ) # [bsz, heads, seq, 64, 2]

        # 展平并拼接
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

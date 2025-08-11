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
该模块实现了不同形式的 Key/Value 缓存结构，包括最基础的全量注意力缓存
以及用于稀疏注意力的 ShadowKV 缓存。通过这些缓存可以在生成式推理中
高效地复用历史状态，减少重复计算与显存开销。
"""

import torch
import math
import gc
from torch import nn
from models.tensor_op import batch_gather_gemm_rotary_pos_emb_cuda
from kernels import shadowkv


class KV_Cache:
    """全量注意力模式下的 KV 缓存管理类"""
    def __init__(self,
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024,
        device :str = 'cuda:0',
        dtype = torch.bfloat16) -> None:

        # 保存模型结构及缓存的基本配置信息
        self.config = config
        self.max_length = max_length               # 支持的最长序列长度
        self.device = device                       # 缓存所在的计算设备
        self.dtype = dtype                         # 缓存张量的数据类型

        # 初始化 Key/Value 缓存，形状为
        # [num_layers, batch, kv_heads, max_length, head_dim]
        # 由于推理过程中长度逐步增加，这里先在 CPU 上分配最大空间
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device='cpu',
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0                         # 当前已缓存的序列长度

        # 用于批量前缀填充的记录指针
        self.prefilled_batch = 0
        self.batch_size = batch_size

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int
            ):
        """
        更新指定层的 Key/Value 缓存
        
        将新计算的 key/value 状态添加到缓存中，并返回当前层的完整缓存
        用于后续的注意力计算。支持批量处理和增量更新。
        
        Args:
            new_k_cache (torch.Tensor): 新的key状态，形状为[bsz, num_kv_heads, seq_len, head_dim]
            new_v_cache (torch.Tensor): 新的value状态，形状为[bsz, num_kv_heads, seq_len, head_dim]
            layer_idx (int): 目标层的索引
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 返回更新后的完整key和value缓存
        """

        # 获取输入张量的维度信息
        bsz, _, incoming, _ = new_v_cache.shape  # [bsz, num_kv_heads, incoming, head_dim]

        # 如果一次性写入完整 batch，则需要重置 prefilled_batch
        if bsz == self.batch_size:
            self.prefilled_batch = 0

        # 将新的 key/value 片段拷贝到缓存的连续位置
        # 使用 prefilled_batch 来跟踪当前批次中已处理的样本数
        self.k_cache[layer_idx][
            self.prefilled_batch:self.prefilled_batch + bsz,
            :,
            self.kv_offset:self.kv_offset + incoming
        ].copy_(new_k_cache)
        self.v_cache[layer_idx][
            self.prefilled_batch:self.prefilled_batch + bsz,
            :,
            self.kv_offset:self.kv_offset + incoming
        ].copy_(new_v_cache)

        # 取出当前层已经缓存的所有 key/value 供后续注意力使用
        key = self.k_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]
        value = self.v_cache[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.kv_offset + incoming]

        # 当 incoming>1 时表示处于前缀阶段，需要将缓存转移到 GPU 上
        if incoming > 1:  # prefill
            key = key.to(self.device)
            value = value.to(self.device)

        # 更新偏移量。只有在所有层都写入后才增加 kv_offset
        # 这确保了所有层的缓存状态保持同步
        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        """打印缓存的状态信息，便于调试"""
        print(f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}")

    def H2D(self):
        """将缓存从 CPU 转移到 GPU，通常在前缀阶段调用"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        """重置缓存状态，开始新的推理"""
        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        """返回当前已缓存的序列长度"""
        return self.kv_offset

class ShadowKVCache:
    """
    ShadowKV 稀疏缓存的 GPU 参考实现
    
    ShadowKV通过稀疏注意力机制大幅减少KV缓存的内存占用，同时保持模型性能。
    主要特点包括：
    1. 基于chunk的稀疏选择策略
    2. SVD分解用于key状态的低秩近似
    3. Landmark机制用于高效的相似性检索
    4. 支持动态的KV缓存管理
    """
    def __init__(self, 
        config :object,
        batch_size :int = 1,
        max_length :int = 32*1024, 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
        ) -> None:
        """
        初始化ShadowKV稀疏缓存
        
        Args:
            config (object): 模型配置对象，包含层数、注意力头数等信息
            batch_size (int): 批次大小，当前实现仅支持batch_size=1
            max_length (int): 支持的最大序列长度，默认32K
            device (str): 计算设备，默认'cuda:0'
            dtype: 张量数据类型，默认torch.bfloat16
            sparse_budget (int): 稀疏预算，即保留的token数量，默认2048
            chunk_size (int): chunk大小，用于分块处理，默认8
            rank (int): SVD分解的秩，用于低秩近似，默认160
        """
        
        # 基础配置信息
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # ShadowKV 参数设置
        self.sparse_budget = int(sparse_budget)  # 稀疏预算：保留的token总数
        self.chunk_size = chunk_size             # chunk大小：分块处理的基本单位
        self.rank = rank                         # SVD分解的秩：控制低秩近似的精度
        self.local_chunk = 4                     # 本地chunk数：保留最近的chunk数量
        self.outlier_chunk = 48                  # 异常值chunk数：用于处理特殊情况

        # 当前实现仅支持单样本推理
        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        # 选中的chunk索引，用于记录哪些chunk被保留在稀疏缓存中
        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        # 完整的value缓存，存储所有的value状态
        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        # 稀疏key缓存缓冲区，只存储选中的key状态
        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,  # 额外的4096用于处理边界情况
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        # 稀疏value缓存缓冲区，与key缓存对应
        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        # 模型结构和状态信息
        self.num_layers = config.num_hidden_layers  # 模型层数
        self.kv_offset = 0                          # 当前缓存的序列长度
        self.prefill = 0                            # 预填充阶段的序列长度
        self.gen_offset = 0                         # 生成阶段的偏移量

        # SVD分解相关的缓存，用于key状态的低秩近似
        self.k_landmark = None      # landmark key状态，用于相似性检索
        self.k_landmark_idx = None  # landmark对应的位置索引
        self.U = None              # SVD分解的U矩阵（左奇异向量）
        self.SV = None             # SVD分解的S*V矩阵（奇异值与右奇异向量的乘积）

        # CUDA流，用于异步内存拷贝操作
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        """
        打印ShadowKV缓存的统计信息
        
        输出包括稀疏预算、chunk大小、SVD秩、已缓存长度等关键参数
        """
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        """
        对输入的Key片段进行SVD分解并缓存
        
        SVD分解用于将高维的key状态压缩为低秩表示，从而减少内存占用
        同时保持足够的信息用于后续的注意力计算。
        
        Args:
            new_k_cache (torch.Tensor): 新的key状态，可能的形状：
                - [bsz, num_kv_heads, seq_len, head_dim] 或
                - [bsz, seq_len, hidden_size]
            layer_idx (int): 当前处理的层索引
        """
        # 处理不同形状的key缓存输入
        # new_k_cache 形状可为 [bsz, 8, prefill, 128] 或 [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # 多头格式：[bsz, num_kv_heads, seq_len, head_dim] --> [bsz, seq_len, hidden_size]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # 已经是合并格式：[bsz, seq_len, hidden_size]
            k_cache = new_k_cache
        
        # 在第一层初始化SVD分解的存储空间
        if layer_idx == 0:
            # 初始化U矩阵（左奇异向量）和SV矩阵（奇异值×右奇异向量）
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        # 执行SVD分解：K = U * S * V^T
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)  # 转置V矩阵以便后续计算
        
        # 保存低秩分解结果
        # U: [bsz, seq_len, rank] - 左奇异向量的前rank列
        self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype))
        
        # SV: [bsz, num_kv_heads, rank, head_dim] - 奇异值与右奇异向量的乘积
        # 这样可以通过 U @ SV 重构原始的key状态
        sv_matrix = torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype)
        self.SV[layer_idx].copy_(sv_matrix.view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2))
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        """
        注册landmark用于后续的相似性检索
        
        Landmark是从每个chunk中提取的代表性key状态，用于快速计算
        query与各个chunk的相似性，从而决定哪些chunk应该被保留。
        
        Args:
            k_landmark (torch.Tensor): landmark key状态，形状为[bsz, num_kv_heads, num_chunks, head_dim]
            k_landmark_idx (torch.Tensor): landmark对应的位置索引，形状为[bsz, num_kv_heads, num_chunks]
            layer_idx (int): 当前处理的层索引
        """
        num_landmarks = k_landmark.shape[-2]
        
        # 在第一层初始化landmark存储空间
        if layer_idx == 0:
            # 初始化landmark key状态存储
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            # 初始化landmark位置索引存储
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        # 保存当前层的landmark信息
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            query: torch.Tensor=None
            ):
        """
        在预填充阶段初始化KV缓存并计算landmark
        
        这是ShadowKV的核心方法之一，负责：
        1. 将完整的value状态存储到CPU缓存
        2. 计算chunk划分和稀疏选择参数
        3. 存储本地chunk到GPU缓存
        4. 计算每个chunk的landmark用于后续检索
        
        Args:
            new_v_cache (torch.Tensor): 新的value状态，形状为[bsz, num_kv_heads, seq_len, head_dim]
            layer_idx (int): 当前处理的层索引
            key_states_roped (torch.Tensor): 应用RoPE后的key状态
            query (torch.Tensor, optional): 查询状态，用于某些优化策略
        """

        incoming = new_v_cache.shape[-2]  # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        self.v_cache_cpu[layer_idx][:, :, :incoming] = new_v_cache.clone()

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        self.select_sets = self.sparse_budget // self.chunk_size
        
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][:, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2)  # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(
            landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1),
            key_states_roped_ctx,
            dim=-1
        )  # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(self.batch_size, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(self.batch_size, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][:, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(self.batch_size, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(self.batch_size, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            self.kv_offset += incoming

    def get_retrieval_position_ids(self, layer_idx, query_states):
        """根据查询向量与 landmark 的相似度选出需要检索的块位置"""
        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        self.incoming_q_len = query_states.shape[-2]  # 当前 query 长度，通常为 1
        chunk_attn = torch.einsum(
            'bhgqd,bhdc->bhgqc',
            query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim),
            self.k_landmark[layer_idx].transpose(2, 3)
        ).squeeze(2) / math.sqrt(128)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn = chunk_attn.sum(dim=-2)  # [bsz, kv_heads, chunks]
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(chunk_attn, dim=-2)
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices

        # 根据选择的块获取其在原序列中的绝对位置
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results)
        self.selected_chunk_idx[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = (
            selected_chunks.unsqueeze(-1) * self.chunk_size +
            torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ).view(self.batch_size, self.num_key_value_heads, -1)

        return position_ids
        
    def get_value_cache(self, layer_idx, position_ids):
        """根据 position_ids 从 CPU 缓存中取出对应的 value"""
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        """根据 position_ids 从 SVD 结果重建并 RoPE 对应的 key"""
        u = self.U[layer_idx]  # [bsz, 128k, rank]
        sv = self.SV[layer_idx]  # [bsz, kv_heads, rank, head_dim]

        index_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, u.size(-1))
        u_expand = u.unsqueeze(1).expand(-1, self.num_key_value_heads, -1, -1)
        U_head = torch.gather(u_expand, 2, index_expanded)

        result = torch.einsum('bhrk,bhkd->bhrd', U_head, sv)
        result = rope_func(result, position_ids)

        self.k_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(result, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def update_kv_cache(self,
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        """在解码阶段追加新的 KV 片段"""

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        """重置所有缓存与状态"""
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.selected_chunk_idx.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0
    
    def H2D(self):
        pass

    def get_kv_len(self):
        return self.kv_offset


class ShadowKVCache_CPU:
    """
    ShadowKV 的 CPU 优化实现，支持更多模型与批量场景
    
    这个类实现了ShadowKV的核心算法，通过以下技术实现长序列高效推理：
    1. 稀疏注意力机制：只保留最重要的KV对，大幅减少内存使用
    2. SVD分解：对Key矩阵进行低秩分解，压缩存储空间
    3. Landmark选择：通过余弦相似度识别重要的chunk作为landmark
    4. CPU-GPU混合存储：将大部分数据存储在CPU内存中，按需传输到GPU
    5. 自定义CUDA kernels：优化稀疏数据的gather和copy操作
    """
    def __init__(self, 
        config :object,           # 模型配置对象，包含层数、注意力头数等信息
        batch_size :int = 1,      # 批处理大小
        max_length :int = 32*1024, # 支持的最大序列长度
        device :str = 'cuda:0',   # GPU设备
        dtype = torch.bfloat16,   # 数据类型，使用bfloat16节省内存
        sparse_budget: int = 2048, # 稀疏预算：保留的KV对数量
        chunk_size=8,             # chunk大小，用于分块处理
        rank=160,                 # SVD分解的秩，控制压缩比例
        ) -> None:
        
        # ========== 基础配置信息 ==========
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        # GQA (Grouped Query Attention) 相关配置
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # 每个KV头对应的Q头数量
        self.head_dim = config.hidden_size // config.num_attention_heads  # 每个注意力头的维度
        self.num_attention_heads = config.num_attention_heads  # Query头的数量
        self.num_key_value_heads = config.num_key_value_heads  # Key-Value头的数量

        # ========== 稀疏注意力配置 ==========
        self.sparse_budget = int(sparse_budget)  # 稀疏预算：每层保留的KV对数量
        self.chunk_size = chunk_size             # chunk大小：将序列分块处理，提高缓存局部性
        self.rank = rank                         # SVD分解的秩：控制Key矩阵的压缩程度
        self.local_chunk = 4                     # 本地chunk数量：保留最近的几个chunk
        self.outlier_chunk = int((self.sparse_budget // 1024) * 24)  # 异常值chunk数量：基于稀疏预算动态计算

        # ========== CPU端Value缓存 ==========
        # 在CPU内存中存储完整的Value缓存，使用pin_memory加速CPU-GPU传输
        # 维度: [层数, 批大小, KV头数, chunk数量, chunk_size*head_dim]
        # 这样的布局便于按chunk进行gather操作
        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,                                              # 模型层数
            batch_size,                                                           # 批处理大小
            config.num_key_value_heads,                                          # KV头数量
            self.max_length // self.chunk_size,                                 # 最大chunk数量
            self.config.hidden_size // self.config.num_attention_heads * self.chunk_size,  # 每个chunk的特征维度
            device='cpu',        # 存储在CPU内存中
            dtype=self.dtype,    # 使用指定的数据类型
            pin_memory=True      # 启用固定内存，加速CPU-GPU数据传输
        )

        # ========== GPU端Key缓存缓冲区 ==========
        # 在GPU内存中存储稀疏的Key缓存，包含：
        # 1. 稀疏预算的KV对 (sparse_budget)
        # 2. 预留空间 (128)
        # 3. 异常值chunk和本地chunk ((outlier_chunk+local_chunk)*chunk_size)
        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,                                              # 模型层数
            batch_size,                                                           # 批处理大小
            config.num_key_value_heads,                                          # KV头数量
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,  # 缓冲区总长度
            self.config.hidden_size // self.config.num_attention_heads,         # 每个头的特征维度
            device=self.device,  # 存储在GPU内存中
            dtype=self.dtype     # 使用指定的数据类型
        )

        # ========== GPU端Value缓存缓冲区 ==========
        # 与Key缓存缓冲区结构相同，存储对应的Value数据
        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,                                              # 模型层数
            batch_size,                                                           # 批处理大小
            config.num_key_value_heads,                                          # KV头数量
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,  # 缓冲区总长度
            self.config.hidden_size // self.config.num_attention_heads,         # 每个头的特征维度
            device=self.device,  # 存储在GPU内存中
            dtype=self.dtype     # 使用指定的数据类型
        )

        # ========== 状态跟踪变量 ==========
        self.num_layers = config.num_hidden_layers  # 模型总层数
        self.kv_offset = 0      # 当前KV缓存的偏移量（已处理的token数量）
        self.prefill = 0        # prefill阶段的序列长度
        self.gen_offset = 0     # generation阶段的偏移量

        # ========== SVD分解和Landmark相关变量 ==========
        # 这些变量在首次使用时初始化，用于存储SVD分解结果和landmark信息
        self.k_landmark = None      # Key的landmark表示：每个chunk的代表性向量
        self.k_landmark_idx = None  # landmark对应的chunk索引
        self.U = None              # SVD分解的U矩阵：左奇异向量
        self.SV = None             # SVD分解的S*V矩阵：奇异值与右奇异向量的乘积

        # ========== 稀疏选择配置 ==========
        self.select_sets = self.sparse_budget // self.chunk_size  # 选择的chunk集合数量
        # 确保稀疏预算能被chunk_size整除，保证内存对齐
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        # ========== 临时缓冲区 ==========
        # 用于CPU-GPU数据传输的临时缓冲区，存储选中的Value chunk
        self.temp = torch.zeros(
            self.batch_size,                    # 批处理大小
            self.num_key_value_heads,          # KV头数量
            self.select_sets,                  # 选择的chunk数量
            self.chunk_size*self.head_dim,     # 每个chunk的特征维度
            device='cpu',                      # 存储在CPU内存中
            dtype=self.dtype                   # 使用指定的数据类型
        ).contiguous()  # 确保内存连续，提高访问效率

        # ========== 批处理状态 ==========
        self.prefilled_batch = 0  # 已完成prefill的批次数量

        # ========== CUDA Kernel相关变量 ==========
        # 用于Value offload kernels的参数
        self.block_num = int(self.batch_size * self.num_key_value_heads)  # 总的处理块数量
        # offsets: 每个块在gather操作中的偏移量数组
        self.offsets = torch.zeros(self.block_num*(sparse_budget // chunk_size), device=self.device, dtype=torch.int32).contiguous()
        # cnts: 每个块需要gather的元素数量
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        # signals: 用于多线程同步的信号量
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        # position_ids: 存储每层每个头选中的chunk位置，初始化为-1表示未选择
        self.position_ids = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # ========== Key计算kernel输出缓冲区 ==========
        # 用于存储SVD重构后的Key数据
        self.output = torch.zeros(
            self.batch_size,           # 批处理大小
            self.num_key_value_heads,  # KV头数量
            sparse_budget,             # 稀疏预算大小
            self.head_dim,             # 头维度
            device='cpu',              # 存储在CPU内存中
            dtype=self.dtype           # 使用指定的数据类型
        ).contiguous()  # 确保内存连续

        # ========== CUDA流管理 ==========
        # 创建专用的CUDA流用于异步数据传输，提高GPU利用率
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        """
        打印ShadowKV缓存的统计信息，用于调试和性能监控
        
        输出信息包括：
        - sparse_budget: 稀疏预算大小
        - chunk_size: chunk大小
        - rank: SVD分解的秩
        - cached: 当前缓存的token数量
        - local_chunk: 本地chunk数量
        - outlier_chunk: 异常值chunk数量
        """
        print(f"ShadowKV_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Encoding阶段：处理prefill输入 #####
    def get_svd(self, new_k_cache, layer_idx):
        """
        对Key缓存进行SVD分解，实现低秩压缩存储
        
        SVD分解的目的：
        1. 将大型Key矩阵分解为U、S、V三个矩阵
        2. 只保留前rank个奇异值，实现压缩
        3. 在推理时通过U和SV重构所需的Key向量
        
        Args:
            new_k_cache: 新的Key缓存，形状为[bsz, kv_heads, seq_len, head_dim]或[bsz, seq_len, hidden_size]
            layer_idx: 当前处理的层索引
        """
        # ========== 数据形状标准化 ==========
        # 处理两种可能的输入格式：
        # 1. [bsz, kv_heads, seq_len, head_dim] - 多头格式
        # 2. [bsz, seq_len, hidden_size] - 扁平格式
        if new_k_cache.shape[1] <= 32:  # 判断是否为多头格式（kv_heads通常小于32）
            # 将多头格式转换为扁平格式：[bsz, kv_heads, seq_len, head_dim] -> [bsz, seq_len, hidden_size]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # 已经是扁平格式，直接使用
            k_cache = new_k_cache
        
        # ========== 初始化SVD存储矩阵 ==========
        # 只在处理第一层第一批次时初始化，避免重复分配内存
        if layer_idx == 0 and self.prefilled_batch == 0:
            # U矩阵：左奇异向量，形状[层数, 批大小, 序列长度, 秩]
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device='cpu', dtype=self.dtype)
            # SV矩阵：奇异值与右奇异向量的乘积，形状[层数, 批大小, KV头数, 头维度, 秩]
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank, device='cpu', dtype=self.dtype)
        
        # ========== 内存清理 ==========
        # SVD计算前清理GPU内存，确保有足够空间进行计算
        torch.cuda.synchronize()  # 等待所有CUDA操作完成
        gc.collect()               # 垃圾回收
        torch.cuda.empty_cache()   # 清空CUDA缓存
        torch.cuda.synchronize()   # 再次同步
        
        # ========== 执行SVD分解 ==========
        # 使用float32精度进行SVD计算，提高数值稳定性
        u, s, v = torch.svd(k_cache.float())  # u: [bsz, seq_len, seq_len], s: [bsz, min(seq_len, hidden_size)], v: [bsz, hidden_size, seq_len]
        v = v.transpose(1,2)  # 转置V矩阵：[bsz, hidden_size, seq_len] -> [bsz, seq_len, hidden_size]
        
        bsz = k_cache.shape[0]
        
        # ========== 存储U矩阵（左奇异向量）==========
        # 只保留前rank个奇异向量，实现降维压缩
        # 形状变化：[bsz, seq_len, seq_len] -> [bsz, seq_len, rank]
        self.U[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(u[:, :, :self.rank].to(self.dtype))
        
        # ========== 计算并存储SV矩阵 ==========
        # 将奇异值S与右奇异向量V相乘：S * V^T
        # 这样在重构时只需要计算 U * (S * V^T)，减少计算量
        temp_sv = torch.matmul(
            torch.diag_embed(s[:, :self.rank]),  # 将奇异值转换为对角矩阵：[bsz, rank] -> [bsz, rank, rank]
            v[:, :self.rank]                     # 取前rank个右奇异向量：[bsz, seq_len, hidden_size] -> [bsz, rank, hidden_size]
        ).to(self.dtype)  # 结果形状：[bsz, rank, hidden_size]
        
        # 重新整理为多头格式：[bsz, rank, hidden_size] -> [bsz, rank, kv_heads, head_dim] -> [bsz, kv_heads, rank, head_dim]
        temp_sv = temp_sv.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ========== 为CUDA kernel优化数据布局 ==========
        # 转置最后两个维度，便于kernel访问：[bsz, kv_heads, rank, head_dim] -> [bsz, kv_heads, head_dim, rank]
        temp_sv = temp_sv.transpose(-1, -2)
        
        # 存储到SV矩阵中
        self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(temp_sv)

        # ========== 清理临时变量 ==========
        # 删除大型临时变量，释放内存
        del u, s, v

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        """
        注册landmark键值和索引
        
        Landmark是ShadowKV中的核心概念，代表序列中最重要的位置。这些位置在稀疏注意力计算中
        会被优先保留，用于与当前query进行相似度计算，从而确定需要检索的其他KV位置。
        
        Args:
            k_landmark: landmark键值数据，形状[bsz, num_heads, num_landmarks, head_dim]
                       包含被选为landmark的Key向量，用于后续的相似度计算
            k_landmark_idx: landmark索引信息，形状[bsz, num_heads, num_landmarks]
                           记录每个landmark在原始序列中的位置索引
            layer_idx: 当前处理的层索引
        
        功能说明：
        - 存储重要的Key向量作为landmark，用于稀疏注意力的相似度计算
        - 保存landmark在原始序列中的位置信息，便于后续的gather操作
        - 初始化融合GEMM kernel所需的输出缓冲区
        """
        # ========== 获取数据维度信息 ==========
        num_landmarks = k_landmark.shape[-2]  # landmark数量
        bsz = k_landmark.shape[0]             # 当前批次大小
        
        # ========== 初始化landmark存储和kernel缓冲区 ==========
        # 只在处理第一层第一批次时初始化，避免重复分配内存
        if layer_idx == 0 and self.prefilled_batch == 0:
            # 初始化landmark Key存储：[层数, 批大小, KV头数, landmark数量, 头维度]
            # 存储在CPU上，使用pinned memory加速CPU-GPU传输
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            # 初始化landmark索引存储：[层数, 批大小, KV头数, landmark数量]
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # ========== 为融合GEMM kernel初始化输出缓冲区 ==========
            # 这些缓冲区用于batch_gemm_softmax kernel的输出存储
            # gemm_o: GEMM操作的输出缓冲区，存储query与landmark的点积结果
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            # softmax_o: Softmax操作的输出缓冲区，存储注意力权重
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            # norm: 用于Softmax计算的归一化因子缓冲区，按256对齐以优化内存访问
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            # sum: 用于Softmax计算的求和缓冲区，与norm具有相同的对齐策略
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        # ========== 存储当前批次的landmark数据 ==========
        # 将landmark Key数据复制到对应的层和批次位置
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        # 将landmark索引数据复制到对应的层和批次位置
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):
        """
        预填充KV缓存
        
        这是ShadowKV的核心方法之一，负责在prefill阶段初始化KV缓存。该方法实现了
        ShadowKV的关键策略：将长序列分为不同类型的chunk，并选择性地保留重要信息。
        
        Args:
            new_v_cache: 新的Value缓存，形状[bsz, num_kv_heads, seq_len, head_dim]
            layer_idx: 当前处理的层索引
            key_states_roped: 应用RoPE后的Key状态，形状[bsz, num_kv_heads, seq_len, head_dim]
            last_query_states: 最后的query状态（可选）
        
        处理流程：
        1. 将序列分割为多个chunk
        2. 保留最后的local chunk（最近的tokens）
        3. 从其余chunk中选择outlier chunk（异常值chunk）
        4. 将剩余chunk的代表向量作为landmark
        5. 将Value数据存储到CPU，Key数据存储到GPU缓冲区
        """
        
        # ========== 获取输入数据维度信息 ==========
        bsz, _, incoming, _ = new_v_cache.shape  # [bsz, num_kv_heads, incoming_seq_len, head_dim]
        self.prefill = incoming  # 记录prefill序列长度
        
        # ========== 计算chunk分割参数 ==========
        max_ctx_chunks = incoming // self.chunk_size  # 总chunk数量
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size  # 对齐到chunk_size的序列长度
        
        # ========== 存储Value缓存到CPU ==========
        # 将Value数据重新整理为chunk格式并存储到CPU，利用CPU的大内存容量
        # 形状变化：[bsz, num_kv_heads, seq_len, head_dim] -> [bsz, num_kv_heads, num_chunks, chunk_size*head_dim]
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(
            new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size*self.head_dim), 
            non_blocking=True  # 异步复制，提高性能
        )

        # ========== 计算chunk分配策略 ==========
        # 序列布局：[context_chunks, local_chunks, remaining]
        # context_chunks: 用于landmark和outlier选择的chunk
        # local_chunks: 保留的最近token chunk
        self.chunks = incoming // self.chunk_size - self.local_chunk  # 可用于稀疏化的chunk数量
        # 确保chunk数量是8的倍数，便于CUDA kernel优化（向量化访问）
        self.chunks = self.chunks - self.chunks % 8
        
        # ========== 存储Local Chunk到GPU缓冲区 ==========
        # Local chunk包含序列末尾的tokens，这些通常是最重要的（最近的上下文）
        self.prefill_local = incoming - self.chunks * self.chunk_size  # local chunks + 对齐到chunk_size的剩余部分
        # 存储Post-RoPE的Key缓存到GPU缓冲区的开头位置
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        # 存储对应的Value缓存到GPU缓冲区的开头位置
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        # ========== 准备Context Chunk用于Outlier检测 ==========
        # 将context部分的Key重新整理为chunk格式，便于后续处理
        # 形状变化：[bsz, kv_heads, chunks*chunk_size, head_dim] -> [bsz, kv_heads, chunks, chunk_size, head_dim]
        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        
        # ========== 计算Landmark候选向量 ==========
        # 每个chunk的平均向量作为该chunk的代表（landmark候选）
        # 形状：[bsz, kv_heads, chunks, head_dim]
        landmark_candidates = key_states_roped_ctx.mean(dim=-2)
        
        # ========== 计算余弦相似度用于Outlier检测 ==========
        # 计算每个chunk内tokens与该chunk平均向量的余弦相似度
        # 相似度低的chunk被认为是outlier（包含异常或重要信息）
        cos_sim = torch.nn.functional.cosine_similarity(
            landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1),  # 扩展landmark到每个token
            key_states_roped_ctx,  # 原始token向量
            dim=-1  # 在head_dim维度计算相似度
        )  # 输出形状：[bsz, kv_heads, chunks, chunk_size]
        
        # ========== 选择Outlier Chunk ==========
        # 选择每个chunk内最小相似度最小的chunk作为outlier
        # 这些chunk包含与平均向量差异最大的信息，通常更重要
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices  # [bsz, kv_heads, outlier_chunk]
    
        # ========== 提取Outlier Chunk数据 ==========
        # 使用gather操作提取被选为outlier的chunk的Key数据
        # 形状变化：[bsz, kv_heads, chunks, chunk_size, head_dim] -> [bsz, kv_heads, outlier_chunk*chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(
            dim=2, 
            index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)
        ).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        # 同样提取outlier chunk的Value数据
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(
            bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim
        ).gather(
            dim=2, 
            index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)
        ).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        # ========== 计算缓冲区布局参数 ==========
        # GPU缓冲区布局：[local_chunk, outlier_chunk, sparse_budget_space]
        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size  # 稀疏区域起始位置
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget  # 稀疏区域结束位置

        # ========== 计算CUDA kernel参数 ==========
        # 这些参数用于后续的gather_copy kernel调用
        self.kernel_offset = self.sparse_start * self.head_dim  # kernel访问的字节偏移量
        self.kernel_stride = self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim  # kernel访问的步长
        
        # ========== 存储Outlier Chunk到GPU缓冲区 ==========
        # 将outlier chunk存储到local chunk之后的位置
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # ========== 过滤并注册Landmark ==========
        # 从landmark候选中排除已被选为outlier的chunk，剩余的作为landmark注册
        # 目标：[bsz, kv_heads, chunks, head_dim] -> [bsz, kv_heads, chunks-outlier_chunk, head_dim]
        
        # 创建所有chunk的索引：[bsz, kv_heads, chunks]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)
        # 创建掩码，标记非outlier的chunk
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)  # 将outlier位置设为False
        # 提取剩余chunk的索引
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # ========== 注册剩余Chunk为Landmark ==========
        # 提取非outlier chunk的landmark候选向量并注册
        landmark_vectors = landmark_candidates.gather(
            dim=2, 
            index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        ).view(bsz, self.num_key_value_heads, -1, self.head_dim)
        self.register_k_landmark(landmark_vectors, rest_idx, layer_idx)

        # ========== 初始稀疏缓存填充 ==========
        # 使用最后的query状态与landmark进行注意力计算，选择最相关的chunk进行初始填充
        
        # 计算query与landmark的注意力分数
        # einsum: 'bhgd,bhcd->bhgc' 表示 [bsz,heads,groups,dim] × [bsz,heads,chunks,dim] -> [bsz,heads,groups,chunks]
        chunk_attn = torch.einsum(
            'bhgd,bhcd->bhgc', 
            last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim),
            self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)
        ) / math.sqrt(128)  # 缩放因子，通常使用sqrt(head_dim)
        
        # 应用softmax获得注意力权重
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        
        # 在group维度上取最大值，得到每个head对每个chunk的最大注意力权重
        chunk_attn, _ = torch.max(chunk_attn, dim=-2)  # [bsz, kv_heads, chunks]
        
        # ========== 选择Top-K相关Chunk ==========
        # 选择注意力权重最高的select_sets个chunk
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices  # [bsz, kv_heads, select_sets]
        
        # 获取选中chunk的原始索引
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(
            last_query_states.device
        ).gather(dim=-1, index=merged_results)  # [bsz, kv_heads, select_sets]
        
        # 存储选中的chunk索引，用于后续检索
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        
        # ========== 数据有效性检查 ==========
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, \
            f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, \
            f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        
        # ========== 生成详细位置索引并填充缓存 ==========
        # 将chunk索引扩展为具体的token位置索引
        # 每个chunk包含chunk_size个连续的token
        position_ids = (
            selected_chunks.unsqueeze(-1) * self.chunk_size + 
            torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ).view(bsz, self.num_key_value_heads, -1)
        
        # 根据位置索引提取对应的Value数据并存储到稀疏缓存区域
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        
        # 同样提取并存储对应的Key数据
        key_ = key_states_roped.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(key_, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            assert self.sparse_budget < incoming
            # self.kv_offset += incoming
            self.prefilled_batch += bsz

            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

                assert torch.any(self.position_ids == -1) == False, f"The cache for offloading is not built correctly, {self.position_ids}"

    ##### Decoding #####
    def get_retrieval_position_ids(self, layer_idx, query_states):
        """
        获取检索位置ID
        
        这是ShadowKV解码阶段的核心方法，负责根据当前query状态动态选择需要检索的KV位置。
        该方法实现了稀疏注意力的关键逻辑：通过与landmark的相似度计算来确定最相关的chunk。
        
        Args:
            layer_idx: 当前处理的层索引
            query_states: 当前query状态，形状[bsz, num_heads, seq_len, head_dim]
        
        处理流程：
        1. 使用融合GEMM+Softmax kernel计算query与landmark的注意力权重
        2. 选择注意力权重最高的chunk
        3. 重新排序并计算偏移量，为后续的gather操作做准备
        """
        # ========== 记录输入query长度 ==========
        # 在解码阶段，通常incoming_q_len = 1（单个新token）
        self.incoming_q_len = query_states.shape[-2]

        # ========== 融合GEMM+Softmax计算 ==========
        # 使用自定义CUDA kernel进行高效的批量GEMM和Softmax计算
        # 这比分别调用PyTorch的GEMM和Softmax操作更高效
        shadowkv.batch_gemm_softmax(
            query_states.contiguous(),                    # Query张量：[bsz, num_heads, seq_len, head_dim]
            self.k_landmark[layer_idx].contiguous(),      # Landmark Key张量：[bsz, kv_heads, chunks, head_dim]
            self.gemm_o,                                  # GEMM输出缓冲区
            self.norm,                                    # Softmax归一化缓冲区
            self.sum,                                     # Softmax求和缓冲区
            self.softmax_o,                               # Softmax输出缓冲区
            self.batch_size * self.num_key_value_heads,   # 批次大小 × KV头数
            self.num_key_value_groups * self.incoming_q_len,  # Query组数 × Query长度
            self.k_landmark[layer_idx].shape[-2],         # Landmark数量（chunk数量）
            self.head_dim,                                # 头维度
            1 / math.sqrt(128),                           # 缩放因子
            0                                             # 附加参数
        )
        
        # ========== 处理GQA（分组查询注意力）==========
        # 如果使用GQA，需要在组维度上取最大值
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(
                self.softmax_o.view(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, -1), 
                dim=-2  # 在group维度上取最大值
            )  # 输出形状：[bsz, kv_heads, chunks]

        # ========== 选择Top-K相关Chunk ==========
        # 根据注意力权重选择最相关的select_sets个chunk
        merged_results = torch.topk(
            chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), 
            k=self.select_sets, 
            dim=-1
        ).indices  # [bsz, kv_heads, select_sets]
        
        # ========== 获取选中Chunk的原始索引 ==========
        # 将topk结果映射回原始chunk索引
        selected_chunks = self.k_landmark_idx[layer_idx].gather(
            dim=-1, 
            index=merged_results
        )  # [bsz, kv_heads, select_sets]
        
        # ========== 重新排序并计算偏移量 ==========
        # 使用CUDA kernel重新排序选中的chunk并计算gather操作所需的偏移量
        # 这为后续的高效数据检索做准备
        shadowkv.reorder_keys_and_compute_offsets(
            self.position_ids[layer_idx],  # 存储的位置ID
            selected_chunks,               # 当前选中的chunk
            self.offsets,                  # 输出：偏移量数组
            self.cnts,                     # 输出：计数数组
            self.batch_size,               # 批次大小
            self.num_key_value_heads,      # KV头数
            self.select_sets               # 选择的chunk数量
        )

        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):
        """
        获取Value缓存
        
        这个方法负责从CPU和GPU缓存中检索Value数据，是ShadowKV解码阶段的关键组件。
        它使用异步CUDA kernel从CPU内存中gather选中的Value chunk，并将其复制到GPU缓存中。
        
        Args:
            layer_idx: 当前处理的层索引
            position_ids: 位置ID，用于确定需要检索的数据位置
        
        Returns:
            检索到的Value缓存，包含稀疏区域和生成区域的数据
        
        处理流程：
        1. 使用gather_copy_with_offsets kernel从CPU异步复制选中的Value chunk到GPU
        2. 计算当前层的生成偏移量
        3. 返回包含稀疏数据和生成数据的完整Value缓存
        """
        # ========== 异步Gather和复制Value数据 ==========
        # 使用自定义CUDA kernel从CPU内存中gather选中的Value chunk
        # 并异步复制到GPU缓存的稀疏区域
        shadowkv.gather_copy_with_offsets(
            self.v_cache_cpu[layer_idx],           # 源：CPU上的Value缓存
            self.v_cache_buffer[layer_idx],        # 目标：GPU上的Value缓存buffer
            self.temp,                             # 临时缓冲区，用于CPU-GPU数据传输
            self.offsets,                          # 偏移量数组，指定gather的起始位置
            self.cnts,                             # 计数数组，指定每个gather操作的数据量
            self.signals,                          # 信号数组，用于同步异步操作
            self.batch_size,                       # 批次大小
            self.num_key_value_heads,              # KV头数
            int(self.max_ctx_chunks_len*self.head_dim),  # CPU缓存中每个头的最大数据长度
            int(self.sparse_budget*self.head_dim),       # GPU稀疏缓存中每个头的数据长度
            self.kernel_offset,                    # CUDA kernel的偏移参数
            self.kernel_stride,                    # CUDA kernel的步长参数
            self.select_sets                       # 选择的chunk数量
        )

        # ========== 计算生成偏移量 ==========
        # 根据当前层是否为最后一层来确定生成区域的偏移量
        # 最后一层使用当前的gen_offset，其他层需要加上incoming_q_len
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        # ========== 返回完整的Value缓存 ==========
        # 返回包含稀疏区域和生成区域的完整Value缓存
        # sparse_end: 稀疏区域结束位置
        # gen_offset: 生成区域当前偏移量
        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        """
        获取Key缓存
        
        这是ShadowKV中最复杂的方法之一，负责重构和检索Key数据。
        它结合了SVD低秩重构、GPU内存gather操作和RoPE位置编码，
        实现了高效的Key缓存检索和位置编码应用。
        
        Args:
            layer_idx: 当前处理的层索引
            position_ids: 位置ID，用于RoPE位置编码
            rope_func: RoPE函数（未使用，保留接口兼容性）
            cos_sin_cache: 预计算的cos/sin缓存，用于RoPE
        
        Returns:
            重构并应用RoPE的Key缓存，包含稀疏区域和生成区域的数据
        
        处理流程：
        1. 获取SVD分解的U和SV矩阵
        2. 使用GPU-to-GPU gather操作收集选中的Key数据
        3. 通过SVD重构恢复完整的Key向量并应用RoPE
        4. 返回完整的Key缓存
        """
        # ========== 获取SVD分解矩阵 ==========
        # U矩阵：左奇异向量，存储在CPU上进行低秩压缩
        # SV矩阵：奇异值×右奇异向量，用于重构原始Key数据
        u = self.U[layer_idx]   # 形状：[bsz, max_seq_len, rank]
        sv = self.SV[layer_idx] # 形状：[bsz, kv_heads, chunks, rank]

        # ========== GPU内存Gather操作 ==========
        # 使用GPU-to-GPU的gather操作收集选中chunk的Key数据
        # 这比CPU-GPU传输更高效，因为Key数据已经在GPU上
        shadowkv.gather_copy_d2d_with_offsets(
            self.k_cache_buffer[layer_idx],        # GPU上的Key缓存buffer
            self.offsets,                          # 偏移量数组
            self.cnts,                             # 计数数组
            self.batch_size,                       # 批次大小
            self.num_key_value_heads,              # KV头数
            int(self.sparse_budget*self.head_dim), # 稀疏缓存大小
            self.kernel_offset,                    # CUDA kernel偏移参数
            self.kernel_stride,                    # CUDA kernel步长参数
            self.select_sets                       # 选择的chunk数量
        )
        
        # ========== SVD重构 + RoPE位置编码 ==========
        # 使用融合kernel同时完成以下操作：
        # 1. 通过U @ SV重构完整的Key向量
        # 2. 应用RoPE旋转位置编码
        # 3. 将结果存储到稀疏缓存区域
        batch_gather_gemm_rotary_pos_emb_cuda(
            u,                                     # U矩阵：左奇异向量
            sv,                                    # SV矩阵：奇异值×右奇异向量
            cos_sin_cache,                         # RoPE的cos/sin预计算缓存
            position_ids,                          # 位置ID，用于RoPE
            self.output,                           # 输出缓冲区
            self.chunk_size,                       # chunk大小
            self.k_cache_buffer[layer_idx],        # Key缓存buffer
            self.sparse_start,                     # 稀疏区域起始位置
            self.sparse_end,                       # 稀疏区域结束位置
            self.cnts                              # 计数数组
        )

        # ========== 计算生成偏移量 ==========
        # 根据当前层是否为最后一层来确定生成区域的偏移量
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        # ========== 返回完整的Key缓存 ==========
        # 返回包含重构的稀疏区域和生成区域的完整Key缓存
        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        """
        Host to Device数据迁移
        
        将关键的缓存数据从CPU（Host）迁移到GPU（Device）。
        这个方法在prefill阶段完成后调用，将计算好的SVD矩阵、landmark数据
        和其他缓冲区从CPU迁移到GPU，为高效的解码阶段做准备。
        
        迁移的数据包括：
        - SVD分解矩阵（U, SV）
        - Landmark数据（k_landmark, k_landmark_idx）
        - GEMM和Softmax缓冲区
        - 临时缓冲区（temp, output）
        
        注意：迁移前后都进行内存清理，确保GPU内存的高效利用。
        """
        # ========== 预清理内存 ==========
        gc.collect()                    # Python垃圾回收
        torch.cuda.empty_cache()        # 清空CUDA缓存
        torch.cuda.synchronize()        # 同步CUDA操作
        
        # ========== 迁移SVD相关数据 ==========
        # 将SVD分解的矩阵从CPU迁移到GPU，用于Key重构
        self.SV = self.SV.to(self.device)                    # 奇异值×右奇异向量矩阵
        self.U = self.U.to(self.device)                      # 左奇异向量矩阵
        
        # ========== 迁移Landmark相关数据 ==========
        # 将landmark数据迁移到GPU，用于稀疏注意力计算
        self.k_landmark = self.k_landmark.to(self.device)         # Landmark Key向量
        self.k_landmark_idx = self.k_landmark_idx.to(self.device) # Landmark索引

        # ========== 迁移GEMM和Softmax缓冲区 ==========
        # 将融合GEMM+Softmax操作的缓冲区迁移到GPU
        self.gemm_o = self.gemm_o.to(self.device)            # GEMM输出缓冲区
        self.softmax_o = self.softmax_o.to(self.device)      # Softmax输出缓冲区
        self.norm = self.norm.to(self.device)                # Softmax归一化缓冲区
        self.sum = self.sum.to(self.device)                  # Softmax求和缓冲区

        # ========== 迁移临时缓冲区 ==========
        # 将临时工作缓冲区迁移到GPU
        self.temp = self.temp.to(self.device)                # CPU-GPU数据传输临时缓冲区
        self.output = self.output.to(self.device)            # SVD重构输出缓冲区

        # ========== 后清理内存 ==========
        torch.cuda.synchronize()        # 确保所有迁移操作完成
        gc.collect()                    # 再次进行垃圾回收
        torch.cuda.empty_cache()        # 清理GPU内存碎片
        torch.cuda.synchronize()        # 最终同步

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        """
        更新KV缓存
        
        在解码阶段，将新生成的Key和Value状态添加到缓存中。
        这个方法负责将新的KV对存储到缓存buffer的生成区域（generation region）。
        
        Args:
            new_k_cache: 新的Key缓存，形状[bsz, num_heads, seq_len, head_dim]
            new_v_cache: 新的Value缓存，形状[bsz, num_heads, seq_len, head_dim]
            layer_idx: 当前处理的层索引
        
        缓存布局：
        [稀疏区域(sparse region) | 生成区域(generation region)]
        - 稀疏区域：存储通过稀疏注意力选择的历史KV对
        - 生成区域：存储解码过程中新生成的KV对
        """
        # ========== 计算新增序列长度 ==========
        incoming = new_k_cache.shape[-2]  # 通常在解码阶段为1
        
        # ========== 更新Value缓存 ==========
        # 将新的Value状态复制到生成区域的对应位置
        # sparse_end: 稀疏区域的结束位置
        # gen_offset: 当前生成区域的偏移量
        self.v_cache_buffer[layer_idx][
            :, :, 
            self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming
        ].copy_(new_v_cache, non_blocking=True)
        
        # ========== 更新Key缓存 ==========
        # 将新的Key状态复制到生成区域的对应位置
        self.k_cache_buffer[layer_idx][
            :, :, 
            self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming
        ].copy_(new_k_cache, non_blocking=True)

        # ========== 更新全局偏移量 ==========
        # 只在处理最后一层时更新全局偏移量，避免重复更新
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming      # 总的KV偏移量
            self.gen_offset += incoming     # 生成区域偏移量

    def clear(self):
        """
        清理缓存状态
        
        重置ShadowKV缓存的所有状态，为新的推理会话做准备。
        这个方法会清零所有缓存数据，释放landmark和SVD相关的内存，
        并重置所有偏移量和状态标志。
        
        清理的内容包括：
        - KV缓存buffer（置零）
        - Landmark数据（释放内存）
        - SVD矩阵（释放内存）
        - 偏移量计数器（重置为0）
        - Prefill状态标志（重置）
        """
        # ========== 清零缓存Buffer ==========
        self.k_cache_buffer.zero_()          # 清零Key缓存buffer
        self.v_cache_buffer.zero_()          # 清零Value缓存buffer
        
        # ========== 释放Landmark相关内存 ==========
        self.k_landmark = None               # 释放landmark Key向量
        self.k_landmark_idx = None           # 释放landmark索引
        
        # ========== 释放SVD相关内存 ==========
        self.U = None                        # 释放左奇异向量矩阵
        self.SV = None                       # 释放奇异值×右奇异向量矩阵

        # ========== 重置偏移量和状态计数器 ==========
        self.kv_offset = 0                   # 重置总KV偏移量
        self.prefill = 0                     # 重置prefill状态
        self.gen_offset = 0                  # 重置生成区域偏移量
        self.prefill_local = 0               # 重置本地prefill计数

        # ========== 重置Prefill批次状态 ==========
        self.prefilled_batch = 0             # 重置prefill批次计数

    def get_kv_len(self):
        """
        获取当前KV缓存长度
        
        返回当前缓存中存储的KV对的总数量。
        这个值等于prefill阶段处理的序列长度加上解码阶段生成的token数量。
        
        Returns:
            int: 当前KV缓存的长度（token数量）
        """
        return self.kv_offset

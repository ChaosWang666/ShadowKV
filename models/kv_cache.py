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

    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
    ) -> None:
        """初始化缓存结构并在 CPU 上预分配空间。

        参数:
            config (object): 包含层数、头数等信息的模型配置对象。
            batch_size (int): 推理时的批大小。
            max_length (int): 允许缓存的最大序列长度。
            device (str): 缓存最终要使用的计算设备。
            dtype (torch.dtype): 缓存张量的数据类型。
        """

        # 保存模型结构及缓存的基本配置信息
        self.config = config
        self.max_length = max_length  # 支持的最长序列长度
        self.device = device  # 缓存所在的计算设备
        self.dtype = dtype  # 缓存张量的数据类型

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

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """追加新的 Key/Value 片段到缓存。

        参数:
            new_k_cache (torch.Tensor): 新生成的 key 张量，形状为
                ``[batch, num_kv_heads, seq_len, head_dim]``。
            new_v_cache (torch.Tensor): 新生成的 value 张量，形状同上。
            layer_idx (int): 当前写入的 Transformer 层索引。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 当前层可用于注意力的 key 和 value
            缓存，形状 ``[batch, num_kv_heads, cache_len, head_dim]``。
        """

        bsz, _, incoming, _ = new_v_cache.shape  # [bsz, num_kv_heads, incoming, head_dim]

        # 如果一次性写入完整 batch，则需要重置 prefilled_batch
        if bsz == self.batch_size:
            self.prefilled_batch = 0

        # 将新的 key/value 片段拷贝到缓存的连续位置
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
        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming

        return key.to(self.device), value.to(self.device)
    
    def print_stats(self):
        """打印缓存的统计信息。"""

        print(
            f"KVCache | max_length {self.max_length} | dtype {self.dtype} | cached {self.kv_offset}"
        )

    def H2D(self):
        """将关键数据从 CPU 迁移到 GPU，以便加速后续计算。"""
        """将缓存从 CPU 转移到 GPU，通常在前缀阶段调用。"""

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.k_cache = self.k_cache.to(self.device)
        self.v_cache = self.v_cache.to(self.device)

    def clear(self):
        """重置缓存状态，以便开始新的推理任务。"""

        self.kv_offset = 0
        self.prefilled_batch = 0

    def get_kv_len(self):
        """返回当前已缓存的序列长度。"""

        return self.kv_offset

class ShadowKVCache:
    """ShadowKV 稀疏缓存的 GPU 参考实现"""

    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
    ) -> None:
        """初始化 ShadowKV 缓存。

        参数:
            config (object): 模型配置对象。
            batch_size (int): 推理批大小，仅支持 1。
            max_length (int): 序列允许的最大长度。
            device (str): 计算设备。
            dtype (torch.dtype): 缓存张量数据类型。
            sparse_budget (int): 稀疏缓存允许保留的 token 数量。
            chunk_size (int): 切分块的大小。
            rank (int): SVD 分解保留的秩。
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
        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        self.local_chunk = 4
        self.outlier_chunk = 48

        assert self.batch_size == 1, "ShadowKV class only supports batch_size=1, please use ShadowKV_CPU class for batch_size > 1"

        self.selected_chunk_idx = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget // self.chunk_size,
            device=self.device,
            dtype=torch.long
        )

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 4096,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )


        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    def get_svd(self, new_k_cache, layer_idx):
        """对输入的 Key 片段做 SVD 分解并缓存。

        参数:
            new_k_cache (torch.Tensor): 未经过 RoPE 的 key 张量，可为
                ``[bsz, 8, prefill, 128]`` 或 ``[bsz, prefill, 1024]`` 形状。
            layer_idx (int): 当前层索引。
        """
        # new_k_cache 形状可为 [bsz, 8, prefill, 128] 或 [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device=self.device, dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.rank, self.head_dim, device=self.device, dtype=self.dtype)
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        self.SV[layer_idx].copy_(torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(self.batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)) # [bsz, 8, 160, 128]
    
    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        """注册 landmark，用于后续检索。

        参数:
            k_landmark (torch.Tensor): landmark 的 key 表示。
            k_landmark_idx (torch.Tensor): 对应的块索引。
            layer_idx (int): 层索引。
        """
        num_landmarks = k_landmark.shape[-2]
        if layer_idx == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device=self.device, dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device=self.device, dtype=torch.long)
        
        self.k_landmark[layer_idx].copy_(k_landmark.contiguous())
        self.k_landmark_idx[layer_idx].copy_(k_landmark_idx.contiguous())

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor | None = None,
    ):
        """在前缀阶段写入初始 KV 并计算 landmark。

        参数:
            new_v_cache (torch.Tensor): 新的 value 缓存片段。
            layer_idx (int): 层索引。
            key_states_roped (torch.Tensor): 已经经过 RoPE 的 key。
            query (torch.Tensor, 可选): 用于计算相似度的查询向量。
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
        """根据查询向量与 landmark 的相似度选出需要检索的块位置。

        参数:
            layer_idx (int): 当前层索引。
            query_states (torch.Tensor): 当前生成步骤的查询向量。

        返回:
            torch.Tensor: 被选中的块对应的位置索引。
        """
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
        """根据给定位置获取对应的 value 片段。

        参数:
            layer_idx (int): 层索引。
            position_ids (torch.Tensor): 待检索的绝对位置。

        返回:
            torch.Tensor: 聚合后的 value 缓存。
        """
        value_ = self.v_cache_cpu[layer_idx].gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][:, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        """根据位置 ID 重建并位置编码对应的 key。

        参数:
            layer_idx (int): 层索引。
            position_ids (torch.Tensor): 需要重建的位置索引。
            rope_func (Callable): RoPE 函数。
            cos_sin_cache (torch.Tensor): RoPE 的 cos/sin 缓存。

        返回:
            torch.Tensor: 重建后的 key 缓存。"""
        """根据 position_ids 从 SVD 结果重建并 RoPE 对应的 key。

        参数:
            layer_idx (int): 层索引。
            position_ids (torch.Tensor): 需要重建的位置索引。
            rope_func (Callable): 应用 RoPE 的函数。
            cos_sin_cache (torch.Tensor): 旋转位置编码所需的 cos/sin 缓存。

        返回:
            torch.Tensor: 重建并位置编码后的 key 缓存。
        """
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

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """在解码阶段追加新的 KV 片段。

        参数:
            new_k_cache (torch.Tensor): 新的 key 张量。
            new_v_cache (torch.Tensor): 新的 value 张量。
            layer_idx (int): 层索引。
        """

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        """重置所有缓存与状态。"""
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
        """将关键数据从 CPU 迁移到 GPU，以便加速后续计算。"""
        """保留接口，与 GPU 版本保持一致。"""

        pass

    def get_kv_len(self):
        """返回当前已缓存的 token 数量。"""

        return self.kv_offset


class ShadowKVCache_CPU:
    """ShadowKV 的 CPU 优化实现，支持更多模型与批量场景"""

    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size=8,
        rank=160,
    ) -> None:
        """初始化 CPU 版本的 ShadowKV 缓存。

        参数:
            config (object): 模型配置对象。
            batch_size (int): 推理批大小。
            max_length (int): 支持的最大上下文长度。
            device (str): 计算设备。
            dtype (torch.dtype): 缓存张量的数据类型。
            sparse_budget (int): 稀疏缓存可保留的 token 数量。
            chunk_size (int): 切分块大小。
            rank (int): SVD 保留的秩。
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

        self.sparse_budget = int(sparse_budget)
        self.chunk_size = chunk_size
        self.rank = rank
        self.local_chunk = 4
        self.outlier_chunk = int((self.sparse_budget // 1024) * 24)

        self.v_cache_cpu = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.max_length // self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads * self.chunk_size,
            device='cpu',
            dtype=self.dtype,
            pin_memory=True
        )

        self.k_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache_buffer = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            self.sparse_budget + 128 + (self.outlier_chunk+self.local_chunk)*self.chunk_size,
            self.config.hidden_size // self.config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0

        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.select_sets = self.sparse_budget // self.chunk_size
        assert self.select_sets * self.chunk_size == self.sparse_budget, f"({self.select_sets}) * {self.chunk_size} != {self.sparse_budget}"

        self.temp = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # batch prefill record
        self.prefilled_batch = 0

        # v offload kernels
        self.block_num = int(self.batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(sparse_budget // chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.position_ids = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.select_sets, device=self.device, dtype=torch.int64).fill_(-1).contiguous()

        # k compute kernels
        self.output = torch.zeros(
            self.batch_size, 
            self.num_key_value_heads, 
            sparse_budget, 
            self.head_dim, 
            device='cpu', 
            dtype=self.dtype
        ).contiguous()

        # multi-stream
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"ShadowKV_CPU | sparse budget {self.sparse_budget} | chunk size {self.chunk_size} |rank {self.rank} | cached {self.kv_offset} | local_chunk {self.local_chunk} | outlier_chunk {self.outlier_chunk}")

    ##### Encoding #####
    def get_svd(self, new_k_cache, layer_idx):
        # [bsz, 8, prefill, 128] OR [bsz, prefill, 1024]
        if new_k_cache.shape[1] <= 32:
            # [bsz, 8, prefill, 128] --> [bsz, prefill, 1024]
            k_cache = new_k_cache.transpose(1, 2).reshape(self.batch_size, -1, self.num_key_value_heads*self.head_dim)
        else:
            # [bsz, prefill, 1024]
            k_cache = new_k_cache
        
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init U, SV
            self.U = torch.zeros(self.num_layers, self.batch_size, k_cache.shape[1], self.rank, device='cpu', dtype=self.dtype)
            self.SV = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, self.head_dim, self.rank, device='cpu', dtype=self.dtype)
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        u, s, v = torch.svd(k_cache.float())
        v = v.transpose(1,2)
        
        bsz = k_cache.shape[0]
        # [bsz, 128k, 1024] --> [bsz, 128k, 160] [bsz, 160, 1024] (bsz, 8, 160, 128)
        self.U[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(u[:, :, :self.rank].to(self.dtype)) # [bsz, 128k, 160]
        
        temp_sv = torch.matmul(torch.diag_embed(s[:, :self.rank]), v[:, :self.rank]).to(self.dtype).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [bsz, 8, 160, 128]

        # used for kernel
        temp_sv = temp_sv.transpose(-1, -2) # [bsz, 8, 128, 160]
        
        self.SV[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(temp_sv) # [bsz, 8, 128, 160]

        del u, s, v

    def register_k_landmark(self, k_landmark, k_landmark_idx, layer_idx):
        """注册 landmark 以便后续检索。

        参数:
            k_landmark (torch.Tensor): landmark 的 key 表示。
            k_landmark_idx (torch.Tensor): 对应的块索引。
            layer_idx (int): 层索引。
        """

        num_landmarks = k_landmark.shape[-2]
        bsz = k_landmark.shape[0]
        if layer_idx == 0 and self.prefilled_batch == 0:
            # init k_landmark, k_landmark_idx
            self.k_landmark = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, self.head_dim, device='cpu', dtype=self.dtype)
            self.k_landmark_idx = torch.zeros(self.num_layers, self.batch_size, self.num_key_value_heads, num_landmarks, device='cpu', dtype=torch.long)

            # for fused gemm kernel
            self.gemm_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.softmax_o = torch.zeros(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, num_landmarks, device='cpu', dtype=torch.bfloat16).contiguous()
            self.norm = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.batch_size*self.num_key_value_heads, self.num_key_value_groups, (num_landmarks + 256 - 1) // 256, device='cpu', dtype=torch.float).contiguous()
        
        self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark)
        self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(k_landmark_idx)

    def prefill_kv_cache(self,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            key_states_roped: torch.Tensor,
            last_query_states=None
            ):
        """在前缀阶段写入初始 KV 并计算 landmark。

        参数:
            new_v_cache (torch.Tensor): 新的 value 片段。
            layer_idx (int): 层索引。
            key_states_roped (torch.Tensor): 已经 RoPE 的 key。
            last_query_states (torch.Tensor, 可选): 用于初始化检索的最后一个查询。
        """

        bsz, _, incoming, _ = new_v_cache.shape # [bsz, num_kv_heads, incoming, head_dim]
        self.prefill = incoming
        max_ctx_chunks = incoming // self.chunk_size
        self.max_ctx_chunks_len = max_ctx_chunks * self.chunk_size
        self.v_cache_cpu[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :max_ctx_chunks].copy_(new_v_cache[:, :, :self.max_ctx_chunks_len].reshape(bsz, self.num_key_value_heads, max_ctx_chunks, self.chunk_size*self.head_dim), non_blocking=True) # [bsz, num_kv_heads, max_ctx_chunks, chunk_size*head_dim]

        # [x0, x1, ...., self.chunks*chunk_size, local_chunk, rest]
        self.chunks = incoming // self.chunk_size - self.local_chunk 
        # ensure self.chunks is even
        self.chunks = self.chunks - self.chunks % 8
        
        # store Post-RoPE k cache <prefill_local> to the cache
        self.prefill_local = incoming - self.chunks * self.chunk_size # local chunks + align to chunk_size
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(key_states_roped[:, :, -self.prefill_local:])
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, :self.prefill_local].copy_(new_v_cache[:, :, -self.prefill_local:])

        key_states_roped_ctx = key_states_roped[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim)
        landmark_candidates = key_states_roped_ctx.mean(dim=-2) # [bsz, kv_heads, chunks, head_dim]
        
        # compute the cos similarity between it and the original key cache
        cos_sim = torch.nn.functional.cosine_similarity(landmark_candidates.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1), key_states_roped_ctx, dim=-1) # [bsz, kv_heads, chunks, chunk_size]
        
        # get the outlier_chunk idx for each head # [bsz, kv_heads, outlier_chunk]
        outlier_chunk_idx = cos_sim.min(dim=-1).values.topk(self.outlier_chunk, largest=False).indices
    
        # [bsz, kv_heads, chunks, chunk_size, head_dim] --gather[bsz, kv_heads, outlier_chunk]-->[bsz, kv_heads, outlier_chunk, chunk_size, head_dim]
        outlier_chunk_k_cache = key_states_roped_ctx.gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)
        
        outlier_chunk_v_cache = new_v_cache[:,:,:self.chunks*self.chunk_size].view(bsz, self.num_key_value_heads, self.chunks, self.chunk_size, self.head_dim).gather(dim=2, index=outlier_chunk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)).view(bsz, self.num_key_value_heads, self.outlier_chunk*self.chunk_size, self.head_dim)

        self.sparse_start = self.prefill_local + self.outlier_chunk*self.chunk_size
        self.sparse_end = self.prefill_local + self.outlier_chunk*self.chunk_size + self.sparse_budget

        self.kernel_offset = self.sparse_start * self.head_dim
        self.kernel_stride = self.v_cache_buffer[layer_idx].shape[-2] * self.head_dim
        
        # store outlier_chunk to the cache
        self.k_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_k_cache)
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.prefill_local:self.sparse_start].copy_(outlier_chunk_v_cache)

        # filter landmark_candidates using outlier_chunk and register the rest to k_landmark
        # [bsz, kv_heads, chunks, head_dim] --> [bsz, kv_heads, chunks - outlier_chunk, head_dim]
        # get rest_idx: [bsz, kv_heads, chunks] --filter--> [bsz, kv_heads, chunks - outlier_chunk]
        all_idx = torch.arange(self.chunks, device=key_states_roped.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1) # [bsz, kv_heads, chunks]
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_chunk_idx, value=False)
        rest_idx = all_idx.masked_select(mask).view(bsz, self.num_key_value_heads, -1)

        # register rest_idxed landmarks to k_landmark
        self.register_k_landmark(landmark_candidates.gather(dim=2, index=rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)).view(bsz, self.num_key_value_heads, -1, self.head_dim), rest_idx, layer_idx)

        # fill cache for the first time
        chunk_attn = torch.einsum('bhgd,bhcd->bhgc', last_query_states.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim), self.k_landmark[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device)) / math.sqrt(128) # [bsz, 8, 4, chunks]
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype)
        chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, 8, chunks]
        merged_results = torch.topk(chunk_attn, k=self.select_sets, dim=-1).indices # [bsz, 8, select_sets(256)]
        selected_chunks = self.k_landmark_idx[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].to(last_query_states.device).gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].copy_(selected_chunks)
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].max() < self.chunks, f"position_ids exceed the max_length {self.position_ids[layer_idx].max()}"
        assert self.position_ids[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz].min() >= 0, f"position_ids exceed the min_length {self.position_ids[layer_idx].min()}"
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, self.num_key_value_heads, -1)
        value_ = new_v_cache.gather(dim=-2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        self.v_cache_buffer[layer_idx][self.prefilled_batch:self.prefilled_batch + bsz, :, self.sparse_start:self.sparse_end].copy_(value_, non_blocking=True)
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
        """根据查询向量选择稀疏检索的块位置。

        参数:
            layer_idx (int): 层索引。
            query_states (torch.Tensor): 当前查询向量。

        返回:
            torch.Tensor: 按块编码的位置 ID。"""

        # self.k_landmark[layer_idx][:, :, :self.chunks] is [bsz, 8, chunks, head_dim]
        # chunk_attn: [bsz, 32, window_size, chunks]
        self.incoming_q_len = query_states.shape[-2] # 1

        # gemm_softmax
        shadowkv.batch_gemm_softmax(
            query_states.contiguous(),
            self.k_landmark[layer_idx].contiguous(),
            self.gemm_o,
            self.norm,
            self.sum,
            self.softmax_o,
            self.batch_size * self.num_key_value_heads,
            self.num_key_value_groups * self.incoming_q_len,
            self.k_landmark[layer_idx].shape[-2],
            self.head_dim,
            1 / math.sqrt(128),
            0
        )
        if self.num_key_value_groups > 1:
            chunk_attn, _ = torch.max(self.softmax_o.view(self.batch_size, self.num_key_value_heads, self.num_key_value_groups, -1), dim=-2) # [bsz, 8, chunks]

        # [bsz, 8, seq] --> [bsz, 8, select_sets(256)]
        merged_results = torch.topk(chunk_attn.view(self.batch_size, self.num_key_value_heads, -1), k=self.select_sets, dim=-1).indices
        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.k_landmark_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]
        shadowkv.reorder_keys_and_compute_offsets(self.position_ids[layer_idx], selected_chunks, self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, self.select_sets)

        return self.position_ids[layer_idx]

    def get_value_cache(self, layer_idx, position_ids):
        """根据给定位置从 CPU 缓存中取出 value。

        参数:
            layer_idx (int): 层索引。
            position_ids (torch.Tensor): 需要检索的绝对位置。

        返回:
            torch.Tensor: 聚合后的 value 缓存。"""

        shadowkv.gather_copy_with_offsets(self.v_cache_cpu[layer_idx], self.v_cache_buffer[layer_idx], self.temp, self.offsets, self.cnts, self.signals, self.batch_size, self.num_key_value_heads, int(self.max_ctx_chunks_len*self.head_dim), int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.v_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def get_key_cache(self, layer_idx, position_ids, rope_func, cos_sin_cache):
        """根据位置 ID 重建并位置编码对应的 key。

        参数:
            layer_idx (int): 层索引。
            position_ids (torch.Tensor): 需要重建的位置索引。
            rope_func (Callable): RoPE 函数。
            cos_sin_cache (torch.Tensor): RoPE 的 cos/sin 缓存。

        返回:
            torch.Tensor: 重建后的 key 缓存。"""

        # gather key cache and rope them
        u = self.U[layer_idx] # [bsz, 128k, rank]
        sv = self.SV[layer_idx] # [bsz, 8, 128, rank]

        shadowkv.gather_copy_d2d_with_offsets(self.k_cache_buffer[layer_idx], self.offsets, self.cnts, self.batch_size, self.num_key_value_heads, int(self.sparse_budget*self.head_dim), self.kernel_offset, self.kernel_stride, self.select_sets)
        batch_gather_gemm_rotary_pos_emb_cuda(u, sv, cos_sin_cache, position_ids, self.output, self.chunk_size, self.k_cache_buffer[layer_idx], self.sparse_start, self.sparse_end, self.cnts)

        gen_offset = self.gen_offset if layer_idx == self.num_layers - 1 else self.gen_offset + self.incoming_q_len

        return self.k_cache_buffer[layer_idx][:, :, :self.sparse_end + gen_offset]

    def H2D(self):
        """将关键数据从 CPU 迁移到 GPU，以便加速后续计算。"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.SV = self.SV.to(self.device)
        self.U = self.U.to(self.device)
        self.k_landmark = self.k_landmark.to(self.device)
        self.k_landmark_idx = self.k_landmark_idx.to(self.device)

        self.gemm_o = self.gemm_o.to(self.device)
        self.softmax_o = self.softmax_o.to(self.device)
        self.norm = self.norm.to(self.device)
        self.sum = self.sum.to(self.device)

        self.temp = self.temp.to(self.device)
        self.output = self.output.to(self.device)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        """在解码阶段追加新的 KV 片段。

        参数:
            new_k_cache (torch.Tensor): 新的 key 张量。
            new_v_cache (torch.Tensor): 新的 value 张量。
            layer_idx (int): 层索引。
        """

        incoming = new_k_cache.shape[-2]
        self.v_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_v_cache, non_blocking=True)
        self.k_cache_buffer[layer_idx][:, :, self.sparse_end+self.gen_offset:self.sparse_end+self.gen_offset+incoming].copy_(new_k_cache, non_blocking=True)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    def clear(self):
        """重置所有缓存与状态。"""
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.k_landmark = None
        self.k_landmark_idx = None
        self.U = None
        self.SV = None

        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.prefill_local = 0

        self.prefilled_batch = 0

    def get_kv_len(self):
        """返回当前已缓存的 token 数量。"""
        return self.kv_offset

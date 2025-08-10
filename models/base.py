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
基础 LLM 抽象类模块

本模块定义了所有大语言模型的基础抽象类LLM，封装了通用的推理流程与KV缓存管理功能。
主要功能包括：
1. 模型推理的统一接口
2. KV缓存的初始化和管理
3. 文本生成（单样本和批量）
4. 注意力机制的计算
5. ShadowKV优化技术的集成

该基类为不同的具体模型（如Llama、GLM、Qwen等）提供统一的实现框架。
"""

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from flash_attn import flash_attn_with_kvcache

from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU

class LLM:
    """
    大语言模型基类
    
    这是所有具体模型实现的基类，提供了统一的推理接口和KV缓存管理功能。
    支持多种注意力模式：full attention、ShadowKV、ShadowKV_CPU等。
    
    主要属性：
        model_name: 模型名称
        attn_mode: 注意力模式（full/shadowkv/shadowkv_cpu）
        max_length: 最大序列长度
        batch_size: 批处理大小
        device: 计算设备
        dtype: 数据类型
        kv_cache: KV缓存对象
    """

    def __str__(self) -> str:
        """
        返回模型的字符串表示
        
        Returns:
            str: 包含模型基本信息和GPU内存使用情况的字符串
        """
        # 计算GPU内存使用情况
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    def init_kv_cache(self, sparse_budget: int, rank: int, chunk_size: int, config):
        """
        初始化KV缓存
        
        根据注意力模式选择合适的KV缓存实现：
        - full: 标准的全量KV缓存
        - shadowkv: GPU版本的ShadowKV稀疏缓存
        - shadowkv_cpu: CPU版本的ShadowKV稀疏缓存
        
        Args:
            sparse_budget (int): 稀疏预算，控制保留的KV对数量
            rank (int): SVD分解的秩，用于低秩近似
            chunk_size (int): 分块大小，用于分块处理
            config: 模型配置对象
        
        Raises:
            ValueError: 当注意力模式无效时抛出异常
        """
        if self.attn_mode == 'full':
            # 使用标准的全量KV缓存
            self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'shadowkv':
            # 使用GPU版本的ShadowKV稀疏缓存
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            # 使用CPU版本的ShadowKV稀疏缓存
            self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        """
        打印KV缓存的统计信息
        
        用于调试和性能分析，显示缓存的使用情况和统计数据。
        """
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        """
        获取位置编码上下文
        
        根据输入序列长度和已缓存的KV长度，生成对应的位置ID。
        
        Args:
            input_ids (torch.LongTensor): 输入的token ID序列，形状为[batch_size, seq_len]
        
        Returns:
            torch.LongTensor: 位置ID序列，形状为[batch_size, seq_len]
        """
        input_len = input_ids.size(1)  # 当前输入序列长度
        past_len = self.kv_cache.get_kv_len()  # 已缓存的序列长度
        # 生成连续的位置ID，从past_len开始到past_len + input_len
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):
        """
        模型前向推理
        
        执行完整的模型前向传播，包括词嵌入、多层Transformer计算和输出投影。
        
        Args:
            input_ids (torch.LongTensor): 输入的token ID序列，形状为[batch_size, seq_len]
            position_ids (torch.LongTensor): 位置ID序列，形状为[batch_size, seq_len]
        
        Returns:
            torch.Tensor: 输出logits，形状为[batch_size, 1, vocab_size]（解码时）或[batch_size, seq_len, vocab_size]（预填充时）
        """
        # 词嵌入层：将token ID转换为词向量
        hidden_states = F.embedding(input_ids, self.embed_tokens)

        # 逐层计算Transformer层
        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        # 最终的层归一化
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        # 如果是预填充阶段（序列长度>16），只保留最后一个token的隐藏状态
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        
        # 输出投影：将隐藏状态映射到词汇表大小的logits
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        """
        预填充阶段
        
        清空KV缓存并处理输入序列，为后续的增量解码做准备。
        这是生成过程的第一阶段，处理完整的输入prompt。
        
        Args:
            input_ids (torch.LongTensor): 输入的token ID序列，形状为[batch_size, seq_len]
        
        Returns:
            torch.Tensor: 输出logits，形状为[batch_size, 1, vocab_size]
        
        Raises:
            AssertionError: 当KV缓存长度与输入序列长度不匹配时抛出异常
        """
        # 清空KV缓存，开始新的生成序列
        self.kv_cache.clear()
        # 执行前向推理
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        # 验证KV缓存长度是否正确
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        """
        连续预填充
        
        在已有KV缓存的基础上继续处理新的输入序列，不清空现有缓存。
        用于处理超长序列或多轮对话的场景。
        
        Args:
            input_ids (torch.LongTensor): 新增的token ID序列，形状为[batch_size, seq_len]
        
        Returns:
            torch.Tensor: 输出logits，形状为[batch_size, 1, vocab_size]
        """
        # 在现有KV缓存基础上继续推理
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        """
        文本编码
        
        将输入文本转换为token ID序列，支持多种模板格式。
        
        Args:
            text (str): 输入文本
            template (str, optional): 模板类型，可选值：
                - 'chat': 聊天模板，用于对话场景
                - 'ctx': 上下文模板
                - 'prefix': 前缀模板
                - None: 不使用模板
            truncation (bool): 是否截断超长序列，默认为False
        
        Returns:
            torch.LongTensor: token ID序列，形状为[1, seq_len]
        
        Raises:
            AssertionError: 当chat模板中发现bos_token_id时抛出异常
        """
        if template == 'chat':
            # 使用聊天模板格式化文本
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            # 验证bos_token_id不应该出现在输入中（聊天模板已包含）
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            # 使用上下文模板
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            # 使用前缀模板
            text = self.prefix_template.format(ctx=text)
        
        # 标准编码流程
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):
        """
        单层Transformer计算
        
        执行单个Transformer层的完整计算，包括自注意力机制和前馈网络。
        根据不同的KV缓存类型采用不同的优化策略。
        
        Args:
            buffer: 层参数缓冲区，包含权重和偏置
            layer_idx (int): 当前层的索引
            hidden_states (torch.FloatTensor): 输入隐藏状态，形状为[batch_size, seq_len, hidden_size]
            position_ids (torch.LongTensor): 位置ID序列，形状为[batch_size, seq_len]
        
        Returns:
            torch.FloatTensor: 输出隐藏状态，形状为[batch_size, seq_len, hidden_size]
        """
        # 保存残差连接的输入
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        # 计算查询、键、值状态（注意力机制的预处理）
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        # 根据KV缓存类型选择不同的注意力计算策略
        if isinstance(self.kv_cache, KV_Cache):
            # 标准KV缓存：全量注意力计算
            # 应用旋转位置编码（RoPE）
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            # 更新KV缓存
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            # 根据序列长度选择注意力计算方式
            if self.minference == True and q_len > 1:
                # 使用MinInference优化的预填充内核
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                # 使用Flash Attention进行高效注意力计算
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU):
            # ShadowKV缓存：稀疏注意力优化
            
            if q_len > 4*1024: # prefill 预填充阶段
                # 对未应用RoPE的key进行SVD分解并保存
                self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
                # 应用旋转位置编码
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                # 预填充KV缓存，保存重要的键值对
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                
                # 选择注意力计算方式
                if self.minference == True:
                    # 使用MinInference优化内核
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    # 使用Flash Attention
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode 解码阶段
                # 应用旋转位置编码到查询和键
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # 更新KV缓存到缓冲区
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                # 获取检索索引（ShadowKV的核心：选择重要的KV对）
                position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

                # 多流并行优化：异步获取value缓存
                curr_stream = torch.cuda.current_stream()
                get_value_stream = self.kv_cache.copy_stream

                with torch.cuda.stream(get_value_stream):
                    get_value_stream.wait_stream(curr_stream)
                    # 异步获取value缓存
                    value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                # 从GPU收集key缓存并应用RoPE（通过CPU卸载时间隐藏延迟）
                key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)

                # 等待value获取完成
                curr_stream.wait_stream(get_value_stream)

                # 执行Flash Attention计算
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        # 重塑注意力输出的形状
        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        # 后注意力计算（前馈网络和残差连接）
        # 对于大批次，使用分块处理以节省内存
        if bsz*q_len > 64*1024: # 当总token数超过64K时分块处理
            output = torch.empty_like(hidden_states)
            # 计算分块参数
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            
            # 分块执行后注意力计算
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            # 直接执行后注意力计算
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        """
        解码token ID序列为文本
        
        Args:
            input_ids (torch.Tensor): token ID序列
            skip_special_tokens (bool): 是否跳过特殊token，默认为False
        
        Returns:
            List[str]: 解码后的文本列表
        """
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """
        单样本文本生成
        
        用于准确性评估，不适用于吞吐量评估。执行自回归文本生成，
        支持多种采样策略和终止条件。
        
        Args:
            input_ids (torch.Tensor): 输入的token ID序列，形状为[1, seq_len]
            gen_len (int): 生成的最大token数量，默认256
            temperature (float): 采样温度，控制随机性，默认0.0（贪婪解码）
            top_p (float): nucleus采样的概率阈值，默认0.9
            top_k (int): top-k采样的k值，默认50
            verbose (bool): 是否实时打印生成的文本，默认False
            benchmark (bool): 是否进行性能基准测试，默认False
            cont (bool): 是否继续之前的生成（不清空KV缓存），默认False
        
        Returns:
            List[str]: 生成的文本列表（单个元素）
        
        Raises:
            ValueError: 当输入长度超过最大长度限制时抛出异常
        """
        # 验证输入类型
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # 预填充阶段：处理输入prompt
        if cont == False:
            # 新的生成序列，检查输入长度
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            # 继续之前的生成，检查总长度
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        
        # 采样第一个生成的token
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        # 初始化生成状态
        n = 0  # 已生成的token数量
        pos = 0  # 用于verbose模式的位置追踪
        generated_ids = []  # 存储生成的token ID
        generated_ids.extend(next_token[0].tolist())
        
        # 将KV缓存从主机内存传输到设备内存（如果需要）
        self.kv_cache.H2D()

        # 开始性能计时（如果启用基准测试）
        if benchmark == True:
            start = time.time()
        
        # 自回归生成循环
        while n < gen_len:
            # 执行单步推理
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            # 采样下一个token
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            
            # 实时显示生成的文本（如果启用verbose模式）
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            # 检查各种终止条件
            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3终止符
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi模型终止符
                break
            if next_token[0] in [151329, 151336, 151338]: # glm模型特殊token
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm终止符
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi模型终止符
                break

        # 完成verbose模式的文本显示
        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        
        # 输出性能统计（如果启用基准测试）
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # 将最后一个token输入模型以更新KV缓存
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 返回解码后的生成文本
        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        """
        批量预填充
        
        对多个输入序列并行执行预填充操作，用于批量推理场景。
        为了内存效率，将大批次分解为更小的子批次进行处理。
        
        Args:
            input_ids (torch.Tensor): 批量输入的token ID序列，形状为[batch_size, seq_len]
            benchmark (bool): 是否进行性能基准测试，默认为False
        
        Returns:
            torch.Tensor: 批量输出logits，形状为[batch_size, 1, vocab_size]
        
        Raises:
            AssertionError: 当批次大小不匹配时抛出异常
            ValueError: 当输入长度超过最大长度限制时抛出异常
        """
        # 清空KV缓存，开始新的批量生成
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        # 验证批次大小
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        # 验证输入长度
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        # 初始化输出logits张量
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        # 根据序列长度动态调整子批次大小
        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8  # 超长序列使用更大的子批次
        else:
            T = 4  # 标准序列使用较小的子批次
        
        # 分批处理以节省内存
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            # 获取当前子批次的输入
            req_input_ids = input_ids[bsz:bsz+T]
            # 执行推理并复制结果
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        
        # 验证KV缓存长度
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):
        """
        GPU预热
        
        通过执行一系列矩阵乘法操作来预热GPU，确保后续的推理操作
        能够达到最佳性能。这有助于消除GPU初始化的延迟。
        """
        # 创建随机张量进行预热计算
        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        
        # 执行100次批量矩阵乘法来预热GPU
        for _ in range(100):
            torch.bmm(a, b)
        
        # 清理预热张量
        del a, b

        print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """
        批量文本生成
        
        用于吞吐量评估的批量生成方法。同时处理多个输入序列，
        最大化GPU利用率和整体吞吐量。
        
        Args:
            input_ids (torch.Tensor): 批量输入的token ID序列，形状为[batch_size, seq_len]
            gen_len (int): 生成的最大token数量，默认256
            temperature (float): 采样温度，控制随机性，默认0.0（贪婪解码）
            top_p (float): nucleus采样的概率阈值，默认-1（禁用）
            top_k (int): top-k采样的k值，默认50
            verbose (bool): 是否显示详细信息，默认False
            benchmark (bool): 是否进行性能基准测试，默认False
            cont (bool): 是否继续之前的生成（不清空KV缓存），默认False
        
        Returns:
            List[str] 或 Tuple[List[str], float]: 
                - 如果benchmark=False: 返回生成的文本列表
                - 如果benchmark=True: 返回(生成的文本列表, 吞吐量)
        
        Raises:
            ValueError: 当输入长度超过最大长度限制时抛出异常
        """
        # 验证输入类型
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # 批量预填充阶段
        if cont == False:
            # 新的生成序列，检查输入长度
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            # 继续之前的生成
            logits = self.prefill_cont(input_ids)
        
        # 采样第一批生成的token
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        # 初始化批量生成状态
        n = 0  # 已生成的token数量
        generated_ids = []  # 存储每步生成的token ID
        generated_ids.append(next_token[:, -1].tolist())
        
        # 准备KV缓存和GPU预热
        self.kv_cache.H2D()
        self.warmup()

        # 开始性能计时（如果启用基准测试）
        if benchmark == True:
            start = time.time()
        
        # 批量自回归生成循环
        while n < gen_len:
            # 执行批量推理
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            # 批量采样下一个token
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())

        # 输出性能统计（如果启用基准测试）
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # 将最后一个token输入模型以更新KV缓存
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 转置生成的ID矩阵以匹配批次维度
        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        # 根据是否启用基准测试返回不同的结果
        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)
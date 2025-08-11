/*
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
*/

/**
 * @file functions.h
 * @brief ShadowKV CUDA 内核函数声明头文件
 * 
 * 本文件包含了 ShadowKV 项目中所有 CUDA 内核函数的声明，
 * 这些函数实现了高效的稀疏注意力计算、KV 缓存管理和位置编码等核心功能。
 * 
 * 主要功能模块：
 * 1. Gather-Copy 操作：实现稀疏注意力中的高效数据收集
 * 2. 旋转位置编码 (RoPE)：多种优化版本的 RoPE 实现
 * 3. 批量矩阵运算：融合的 GEMM 和 Softmax 操作
 */

#include <torch/extension.h>

/**
 * @brief 基础的 Gather-Copy 操作
 * 
 * 从 CPU 内存中根据位置 ID 收集 value 数据到 GPU 缓存中，
 * 这是 ShadowKV 稀疏注意力的核心操作之一。
 * 
 * @param values CPU 上的 value 张量
 * @param v_cache_buffer GPU 上的 value 缓存缓冲区
 * @param position_ids 位置 ID 张量，指定要收集的数据位置
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param cpu_v_length CPU 上 value 的长度
 * @param gpu_v_length GPU 上 value 缓存的长度
 * @param map_size 映射表大小
 */
void gather_copy(
    torch::Tensor values,           // 输入：CPU 上的 value 张量 [batch_size, heads, cpu_v_length, embed_dim]
    torch::Tensor v_cache_buffer,   // 输出：GPU 上的 value 缓存缓冲区 [batch_size, heads, gpu_v_length, embed_dim]
    torch::Tensor position_ids,     // 输入：位置 ID 张量，指定要收集的数据位置 [map_size]
    int batch_size,                 // 批次大小
    int heads,                      // 注意力头数
    int cpu_v_length,              // CPU 上 value 的序列长度
    int gpu_v_length,              // GPU 上 value 缓存的序列长度
    int map_size);                 // 映射表大小，即要收集的元素数量

/**
 * @brief GPU 到 GPU 的带偏移量 Gather-Copy 操作（用于 keys）
 * 
 * 在 GPU 内存中直接进行 key 张量的 gather-copy 操作，
 * 使用预计算的偏移量来优化内存访问模式。
 * 
 * @param keys GPU 上的 key 张量
 * @param offsets 输入，由 reorder_keys_and_compute_offsets 计算的偏移量数组
 * @param cnts 输入，由 reorder_keys_and_compute_offsets 计算的计数数组
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param gpu_k_length GPU 上 key 的长度
 * @param gpu_k_offset GPU key 的起始偏移量
 * @param gpu_k_stride GPU key 的步长
 * @param map_size 映射表大小
 */
void gather_copy_d2d_with_offsets(
    torch::Tensor keys,             // 输入：GPU 上的 key 张量 [batch_size, heads, gpu_k_length, embed_dim]
    torch::Tensor offsets,          // 输入：预计算的偏移量数组 [numBlocks*256]，由 reorder_keys_and_compute_offsets 计算
    torch::Tensor cnts,             // 输入：计数数组 [numBlocks]，用于分离 D2D 和 H2D 操作
    int batch_size,                 // 批次大小
    int heads,                      // 注意力头数
    int gpu_k_length,              // GPU key 缓存的序列长度
    int gpu_k_offset,              // GPU key 缓存的起始偏移量
    int gpu_k_stride,              // GPU key 缓存的内存步长
    int map_size);                 // 映射表大小，即要收集的元素数量

/**
 * @brief 重排序 keys 并计算偏移量
 * 
 * 为后续的 gather-copy 操作预处理位置 ID，重排序缓存的位置 ID
 * 并计算相应的偏移量和计数，优化内存访问模式。这是 ShadowKV
 * 稀疏注意力中关键的预处理步骤，确保高效的数据收集。
 * 
 * @param cached_pos_ids 缓存的位置 ID [map_size]（输入输出，int64_t 类型）
 * @param cur_pos_ids 当前的位置 ID [map_size]（输入，int64_t 类型）
 * @param offsets 输出的偏移量数组 [numBlocks]，用于 gather_copy_with_offsets
 * @param cnts 输出的计数数组 [numBlocks]，用于分离 D2D 和 H2D 操作
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param map_size 映射表大小
 */
void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // 输入输出：缓存的位置 ID，同时作为重排序后的位置 ID 输出 (int64_t)
    torch::Tensor cur_pos_ids,    // 输入：当前传入的位置 ID (int64_t)
    torch::Tensor offsets,        // 输出：用于 gather_copy_with_offsets 的偏移量数组 [numBlocks]
    torch::Tensor cnts,           // 输出：用于分离 D2D 和 H2D 操作的计数数组 [numBlocks]
    int batch_size, int heads, int map_size);

/**
 * @brief 带偏移量的 Gather-Copy 操作
 * 
 * 使用预计算的偏移量进行高效的 value 数据收集，支持 CPU 到 GPU
 * 的异步数据传输，是 ShadowKV 中最重要的内存优化操作之一。
 * 该函数实现了稀疏注意力中的核心数据收集逻辑。
 * 
 * @param values CPU 上的 value 张量 [batch_size, heads, cpu_v_length, embed_dim]
 * @param v_cache_buffer GPU 上的 value 缓存缓冲区 [batch_size, heads, gpu_v_length, embed_dim]
 * @param temp 临时 GPU 内存，大小与单层 v_cache_buffer 相同
 * @param offsets 预计算的偏移量数组 [numBlocks]，由 reorder_keys_and_compute_offsets 计算
 * @param cnts 计数数组 [numBlocks]，由 reorder_keys_and_compute_offsets 计算
 * @param signals 内部信号数组 [numBlocks]，全零初始化
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param cpu_v_length CPU 上 value 的序列长度
 * @param gpu_v_length GPU 上 value 缓存的序列长度
 * @param gpu_v_offset GPU value 缓存的起始偏移量
 * @param gpu_v_stride GPU value 缓存的内存步长
 * @param map_size 映射表大小，即要收集的元素数量
 */
void gather_copy_with_offsets(
    torch::Tensor values,           // 输入：CPU 上的 value 张量
    torch::Tensor v_cache_buffer,   // 输入输出：GPU 上的 value 缓存缓冲区
    torch::Tensor temp,             // 临时 GPU 内存，大小与单层 v_cache_buffer 相同
    torch::Tensor offsets,          // 输入：预计算的偏移量数组 [numBlocks]
    torch::Tensor cnts,             // 输入：计数数组 [numBlocks]
    torch::Tensor signals,          // 内部信号数组，全零初始化 [numBlocks]
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size);

/**
 * @brief 应用旋转位置编码 (RoPE) - 基础版本
 * 
 * 基础版本的 RoPE 实现，使用分离的 cos 和 sin 张量来应用旋转位置编码。
 * RoPE 是一种相对位置编码方法，通过旋转变换将位置信息编码到特征中，
 * 在长序列建模中表现优异。该函数是标准的 RoPE 实现。
 * 
 * @param x 输入张量 [batch_size, heads, seq_len, embed_dim]（query 或 key）
 * @param cos 余弦值张量，包含预计算的 cos 值
 * @param sin 正弦值张量，包含预计算的 sin 值
 * @param position_ids 位置 ID 张量，指定每个 token 的位置
 * @param output 输出张量，存储应用 RoPE 后的结果
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb x 张量的批次维度步长
 * @param stride_xh x 张量的头维度步长
 * @param stride_xs x 张量的序列维度步长
 * @param stride_xe x 张量的嵌入维度步长
 * @param stride_cos cos 张量的步长
 * @param stride_sin sin 张量的步长
 * @param stride_pid_b position_ids 的批次维度步长
 * @param stride_pid_h position_ids 的头维度步长
 * @param stride_pid_s position_ids 的序列维度步长
 * @param half_dim 嵌入维度的一半，通常为 embed_dim // 2
 */
void apply_rotary_pos_emb(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos, int stride_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim);

/**
 * @brief 应用旋转位置编码 (RoPE) - 优化版本
 * 
 * 优化版本的 RoPE 实现，使用融合的 cos_sin 张量来减少内存访问
 * 和提高计算效率。相比基础版本，该实现通过合并 cos 和 sin 张量
 * 减少了内存带宽需求，提升了 GPU 利用率和整体性能。
 * 
 * @param x 输入张量 [batch_size, heads, seq_len, embed_dim]（query 或 key）
 * @param cos_sin 融合的余弦正弦张量，包含预计算的 cos 和 sin 值
 * @param position_ids 位置 ID 张量，指定每个 token 的位置
 * @param output 输出张量，存储应用 RoPE 后的结果
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb x 张量的批次维度步长
 * @param stride_xh x 张量的头维度步长
 * @param stride_xs x 张量的序列维度步长
 * @param stride_xe x 张量的嵌入维度步长
 * @param stride_cos_sin cos_sin 张量的步长
 * @param stride_pid_b position_ids 的批次维度步长
 * @param stride_pid_h position_ids 的头维度步长
 * @param stride_pid_s position_ids 的序列维度步长
 * @param half_dim 嵌入维度的一半，通常为 embed_dim // 2
 */
void apply_rotary_pos_emb_new(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim);

/**
 * @brief 旋转位置编码应用 v2 版本
 * 
 * 进一步优化的 RoPE 实现，支持分块处理来提高并行度和缓存效率，
 * 特别适用于长序列的处理。该版本通过分块策略减少了内存占用，
 * 提高了 GPU 的计算效率和吞吐量。
 * 
 * @param x 输入张量 [batch_size, heads, seq_len, embed_dim]（query 或 key）
 * @param cos_sin 融合的余弦和正弦值张量
 * @param position_ids 位置 ID 张量
 * @param output 输出张量
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb x 张量的批次维度步长
 * @param stride_xh x 张量的头维度步长
 * @param stride_xs x 张量的序列维度步长
 * @param stride_xe x 张量的嵌入维度步长
 * @param stride_cos_sin 融合 cos_sin 张量的步长
 * @param stride_pid_b position_ids 的批次维度步长
 * @param stride_pid_h position_ids 的头维度步长
 * @param stride_pid_s position_ids 的序列维度步长
 * @param half_dim 嵌入维度的一半，通常为 embed_dim // 2
 * @param chunk_size 分块大小，用于优化内存访问模式
 */
void apply_rotary_pos_emb_new_v2(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim, int chunk_size);

/**
 * @brief 带缓存推送的旋转位置编码应用
 * 
 * 在应用 RoPE 的同时将结果推送到 KV 缓存中，
 * 用于增量推理场景，避免重复计算和内存拷贝。
 * 
 * @param x 输入张量（query 或 key）
 * @param cos_sin 融合的余弦和正弦值张量
 * @param position_ids 位置 ID 张量
 * @param output 输出张量
 * @param cnts 计数张量，用于控制缓存推送
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb, stride_xh, stride_xs, stride_xe 输入张量的各维度步长
 * @param stride_cos_sin 融合 cos_sin 张量的步长
 * @param stride_pid_b, stride_pid_h, stride_pid_s 位置 ID 张量的各维度步长
 * @param stride_output_b, stride_output_h, stride_output_s 输出张量的各维度步长
 * @param offset_output_s_start, offset_output_s_end 输出序列的起始和结束偏移量
 * @param half_dim 嵌入维度的一半
 * @param chunk_size 分块大小
 */
void apply_rotary_pos_emb_push_cache(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

/**
 * @brief 优化的带缓存推送旋转位置编码应用
 * 
 * 进一步优化的缓存推送版本，通过改进的内存访问模式和计算策略
 * 来提高性能，特别适用于高吞吐量的推理场景。
 * 
 * @param x 输入张量（query 或 key）
 * @param cos_sin 融合的余弦和正弦值张量
 * @param position_ids 位置 ID 张量
 * @param output 输出张量
 * @param cnts 计数张量，用于控制缓存推送
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb, stride_xh, stride_xs, stride_xe 输入张量的各维度步长
 * @param stride_cos_sin 融合 cos_sin 张量的步长
 * @param stride_pid_b, stride_pid_h, stride_pid_s 位置 ID 张量的各维度步长
 * @param stride_output_b, stride_output_h, stride_output_s 输出张量的各维度步长
 * @param offset_output_s_start, offset_output_s_end 输出序列的起始和结束偏移量
 * @param half_dim 嵌入维度的一半
 * @param chunk_size 分块大小
 */
void apply_rotary_pos_emb_push_cache_opt(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

/**
 * @brief GLM 模型专用的优化缓存推送旋转位置编码应用
 * 
 * 专门为 GLM 模型架构优化的 RoPE 实现，考虑了 GLM 特有的
 * 位置编码方式和注意力机制，提供最佳的性能表现。
 * 
 * @param x 输入张量（query 或 key）
 * @param cos_sin 融合的余弦和正弦值张量
 * @param position_ids 位置 ID 张量
 * @param output 输出张量
 * @param cnts 计数张量，用于控制缓存推送
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param stride_xb, stride_xh, stride_xs, stride_xe 输入张量的各维度步长
 * @param stride_cos_sin 融合 cos_sin 张量的步长
 * @param stride_pid_b, stride_pid_h, stride_pid_s 位置 ID 张量的各维度步长
 * @param stride_output_b, stride_output_h, stride_output_s 输出张量的各维度步长
 * @param offset_output_s_start, offset_output_s_end 输出序列的起始和结束偏移量
 * @param half_dim 嵌入维度的一半
 * @param chunk_size 分块大小
 */
void apply_rotary_pos_emb_push_cache_opt_glm(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

/**
 * @brief 批量 Gather GEMM 操作
 * 
 * 结合稀疏注意力的批量矩阵乘法操作，同时集成了 RoPE 应用。
 * 这是 ShadowKV 稀疏注意力计算的核心内核，通过 gather 操作
 * 只计算重要的注意力权重，大幅减少计算量。该内核实现了 SVD 重构、
 * 稀疏收集和矩阵乘法的融合操作，大幅提升了计算效率。
 * 
 * @param a 输入矩阵 A，通常是 query 张量
 * @param b 输入矩阵 B，通常是 key 张量
 * @param cos 余弦值张量，用于 RoPE 计算
 * @param sin 正弦值张量，用于 RoPE 计算
 * @param position_ids 位置 ID 张量
 * @param output 输出张量，存储 GEMM 结果
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param rank SVD 分解的秩，用于低秩近似
 * @param sparse_budget 稀疏预算，控制稀疏度
 * @param max_seq_len 最大序列长度
 * @param chunk_size 分块大小，用于内存优化
 * @param offset_array 偏移量数组，用于稀疏索引
 */
void batch_gather_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor position_ids,
    torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim, int rank, int sparse_budget,
    int max_seq_len, int chunk_size, torch::Tensor offset_array);

/**
 * @brief 批量 GEMM 与 Softmax 融合操作
 * 
 * 将批量矩阵乘法和 Softmax 操作融合在一个内核中，
 * 减少内存访问和提高计算效率，特别适用于注意力权重的计算。
 * 该内核将 GEMM 和 Softmax 融合在一起，减少了中间结果的内存占用
 * 和数据传输开销，是 ShadowKV 稀疏注意力计算的关键优化。
 * 
 * @param A 输入矩阵 A [batch_count, m, k]，通常是 query 张量
 * @param B 输入矩阵 B [batch_count, k, n]，通常是 key 张量
 * @param D 输出矩阵 D [batch_count, m, n]，存储 GEMM 结果
 * @param Norm 归一化因子张量，用于 Softmax 计算
 * @param Sum 求和张量，用于 Softmax 归一化
 * @param Softmax 输出的 Softmax 结果 [batch_count, m, n]，最终的注意力权重
 * @param batch_count 批次数量，即要处理的矩阵对数量
 * @param m 矩阵维度 m，通常对应查询序列长度
 * @param n 矩阵维度 n，通常对应键序列长度
 * @param k 矩阵维度 k，通常对应嵌入维度
 * @param alpha GEMM 操作的缩放因子 alpha
 * @param beta GEMM 操作的缩放因子 beta
 */
void batch_gemm_softmax(torch::Tensor A, torch::Tensor B,
                        torch::Tensor D, torch::Tensor Norm, torch::Tensor Sum,
                        torch::Tensor Softmax, int batch_count, int m, int n,
                        int k, float alpha, float beta);
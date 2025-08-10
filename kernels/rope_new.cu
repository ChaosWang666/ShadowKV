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
 * @file rope_new.cu
 * @brief 优化版旋转位置编码 (RoPE) 的 CUDA 内核实现
 * 
 * 本文件实现了多个优化版本的 RoPE 内核，相比基础版本具有以下改进：
 * 
 * 1. **内存优化**：使用融合的 cos_sin 张量减少内存访问
 * 2. **分块处理**：支持 chunk_size 参数，优化长序列处理
 * 3. **缓存推送**：直接将结果写入 KV 缓存，减少中间拷贝
 * 4. **批量优化**：通过内核内循环减少启动开销
 * 5. **模型特化**：为 GLM 模型提供专门的优化版本
 * 
 * 主要变体：
 * - apply_rotary_pos_emb_new: 基础优化版本（融合 cos_sin）
 * - apply_rotary_pos_emb_new_v2: 支持分块处理的版本
 * - apply_rotary_pos_emb_push_cache: 直接推送到缓存的版本
 * - apply_rotary_pos_emb_push_cache_opt: 进一步优化的缓存推送版本
 * - apply_rotary_pos_emb_push_cache_opt_glm: GLM 专用优化版本
 */

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <vector>
#include "functions.h"

/**
 * @brief 优化版 RoPE 内核 - 使用融合的 cos_sin 张量
 * 
 * 相比基础版本的主要优化：
 * 1. 使用单个 cos_sin 张量替代分离的 cos 和 sin 张量
 * 2. 减少内存访问次数，提高缓存命中率
 * 3. 简化内存布局：[cos_values, sin_values] 连续存储
 * 
 * @param cos_sin 融合的余弦正弦值张量，布局为 [cos_part, sin_part]
 */
__global__ void apply_rotary_pos_emb_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int64_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    // 获取位置 ID 并计算 cos_sin 指针
    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + s_idx * stride_pid_s];
    const __nv_bfloat16* cos_sin_ptr = cos_sin + pid * stride_cos_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        // 从融合张量中读取 cos 和 sin 值
        __nv_bfloat16 cos = cos_sin_ptr[tid];              // cos 部分
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];   // sin 部分

        // 应用旋转变换
        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



/**
 * @brief 优化版 RoPE 的主机端接口
 * 
 * 使用融合的 cos_sin 张量，减少内存带宽需求。
 * 适用于标准的 RoPE 应用场景。
 */
void apply_rotary_pos_emb_new(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        half_dim
    );
}

/**
 * @brief 支持分块处理的 RoPE 内核 v2
 * 
 * 主要改进：
 * 1. 引入 chunk_size 参数，支持分块处理长序列
 * 2. 优化位置 ID 的计算方式：pid = position_ids[s_idx / chunk_size]
 * 3. 改进 cos_sin 索引：(pid * chunk_size + s_idx % chunk_size)
 * 4. 适用于长序列推理和分块注意力机制
 * 
 * @param chunk_size 分块大小，用于长序列的分块处理
 */
__global__ void apply_rotary_pos_emb_kernel_v2(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim, int chunk_size)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    // 分块处理：每个 chunk 共享相同的基础位置 ID
    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
    // 计算当前位置在 chunk 内的偏移
    const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos = cos_sin_ptr[tid];
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



/**
 * @brief 分块处理版 RoPE 的主机端接口
 * 
 * 支持长序列的分块处理，通过 chunk_size 参数控制分块大小。
 * 适用于需要处理超长序列的场景，如长文档理解。
 * 
 * @param chunk_size 分块大小，影响位置编码的计算方式
 */
void apply_rotary_pos_emb_new_v2(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim, int chunk_size)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel_v2<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        half_dim, chunk_size
    );
}

/**
 * @brief 直接推送到 KV 缓存的 RoPE 内核
 * 
 * 主要特性：
 * 1. 应用 RoPE 变换后直接写入 KV 缓存，避免中间拷贝
 * 2. 支持增量推理：通过 cnts 参数跳过已处理的 token
 * 3. 缓存边界检查：确保不超出缓存容量
 * 4. 内存布局优化：支持自定义的缓存步长
 * 
 * @param output_cache KV 缓存张量
 * @param cnts 每个 (batch, head) 已缓存的 token 数量
 * @param offset_output_s_start 缓存写入的起始偏移
 * @param offset_output_s_end 缓存写入的结束偏移
 */
__global__ void apply_rotary_pos_emb_kernel_push_cache(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.z;
    int tid = threadIdx.x;

    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
    // 检查是否已经处理过该 token（增量推理优化）
    int cnt = cnts[b_idx * heads + h_idx];
    if (s_idx / chunk_size < cnt) {
        return;  // 跳过已处理的 token
    }
    const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

    // 计算输入和缓存输出的内存偏移
    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output_cache + out_offset;

    // 缓存边界检查
    if (offset_output_s_start + s_idx >= offset_output_s_end) {
        return;  // 超出缓存容量，跳过
    }

    if (tid < half_dim) {
        __nv_bfloat16 x1 = x_ptr[tid];
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];
        __nv_bfloat16 cos = cos_sin_ptr[tid];
        __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

        output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
    }
}



/**
 * @brief 缓存推送版 RoPE 的主机端接口
 * 
 * 专为增量推理设计，直接将 RoPE 结果写入 KV 缓存。
 * 支持动态缓存管理和边界检查。
 * 
 * @param output_cache KV 缓存张量
 * @param cnts 缓存计数器，记录每个头已缓存的 token 数
 */
void apply_rotary_pos_emb_push_cache(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    const dim3 blocks(batch_size, heads, seq_len);
    const dim3 threads(half_dim);

    apply_rotary_pos_emb_kernel_push_cache<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        stride_output_b, stride_output_h, stride_output_s,
        offset_output_s_start, offset_output_s_end,
        half_dim, chunk_size
    );
}

/**
 * @brief 高度优化的缓存推送 RoPE 内核
 * 
 * 关键优化策略：
 * 1. **重新组织线程布局**：h_idx = threadIdx.x / half_dim, s_idx = blockIdx.x
 * 2. **内核内批量循环**：使用 #pragma unroll 展开批量维度循环
 * 3. **减少内核启动开销**：单次启动处理所有批次
 * 4. **改进内存访问模式**：更好的缓存局部性
 * 5. **线程块配置优化**：seq_len × (heads × half_dim) 的线程配置
 * 
 * 适用场景：高吞吐量推理，批量大小较小但需要高效处理的场景
 */
__global__ void apply_rotary_pos_emb_kernel_push_cache_opt(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    // 重新组织线程索引以优化内存访问
    int h_idx = threadIdx.x / half_dim;  // 注意力头索引
    int s_idx = blockIdx.x;              // 序列位置索引
    int tid = threadIdx.x % half_dim;    // 嵌入维度索引

    // 内核内批量循环，减少启动开销
    #pragma unroll
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        int cnt = cnts[b_idx * heads + h_idx];
        if (s_idx / chunk_size < cnt) {
            continue;
        }

        int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
        const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

        int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
        int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
        const __nv_bfloat16* x_ptr = x + x_offset;
        __nv_bfloat16* output_ptr = output_cache + out_offset;

        if (offset_output_s_start + s_idx >= offset_output_s_end) {
            return;
        }

        if (tid < half_dim) {
            __nv_bfloat16 x1 = x_ptr[tid];
            __nv_bfloat16 x2 = x_ptr[tid + half_dim];
            __nv_bfloat16 cos = cos_sin_ptr[tid];
            __nv_bfloat16 sin = cos_sin_ptr[tid + half_dim];

            output_ptr[tid] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
            output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
        }
    }
}



/**
 * @brief 高度优化的缓存推送 RoPE 主机端接口
 * 
 * 使用优化的线程配置和内核内批量循环，显著提升性能。
 * 线程配置：seq_len 个线程块，每个线程块 heads × half_dim 个线程。
 * 
 * 性能优势：
 * - 减少内核启动开销
 * - 改善内存访问模式
 * - 更好的 GPU 资源利用率
 */
void apply_rotary_pos_emb_push_cache_opt(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    // 优化的线程配置：seq_len 个线程块，每个线程块 heads × half_dim 个线程
    apply_rotary_pos_emb_kernel_push_cache_opt<<<seq_len, heads * half_dim>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        stride_output_b, stride_output_h, stride_output_s,
        offset_output_s_start, offset_output_s_end,
        half_dim, chunk_size
    );
}


/**
 * @brief GLM 模型专用的优化缓存推送 RoPE 内核
 * 
 * GLM 特定优化：
 * 1. **部分 RoPE 应用**：只对前 64 维中的前 32 维应用 RoPE
 * 2. **交错模式**：对 (even_idx, odd_idx) 对应用旋转，而非连续的两半
 * 3. **维度保持**：后 32 维保持不变，直接拷贝
 * 4. **硬编码优化**：针对 GLM 的 half_dim=64 进行特化
 * 
 * GLM RoPE 模式：
 * - 前 32 维：应用交错 RoPE (0,1), (2,3), ..., (62,63)
 * - 后 32 维：保持原值不变
 * 
 * 注意：half_dim 固定为 64，这是 GLM 模型的特定要求
 */
__global__ void apply_rotary_pos_emb_kernel_push_cache_opt_glm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int32_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output_cache,
    const int32_t* __restrict__ cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    int h_idx = threadIdx.x / half_dim;
    int s_idx = blockIdx.x;

    // GLM 模型固定：half_dim = 64
    int tid = threadIdx.x % half_dim;

    // In-kernel loop for batch size with unrolling
    #pragma unroll
    for (int b_idx = 0; b_idx < batch_size; b_idx++) {
        int cnt = cnts[b_idx * heads + h_idx];
        if (s_idx / chunk_size < cnt) {
            continue;
        }

        int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + (s_idx / chunk_size) * stride_pid_s];
        const __nv_bfloat16* cos_sin_ptr = cos_sin + (pid * chunk_size + s_idx % chunk_size) * stride_cos_sin;

        int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
        int out_offset = b_idx * stride_output_b + h_idx * stride_output_h + (offset_output_s_start + s_idx) * stride_output_s;
        const __nv_bfloat16* x_ptr = x + x_offset;
        __nv_bfloat16* output_ptr = output_cache + out_offset;

        if (offset_output_s_start + s_idx >= offset_output_s_end) {
            return;
        }

        // GLM 特定的 RoPE 应用模式
        if (tid < half_dim) {
            if (tid < 32) { // 前 32 个线程处理需要 RoPE 的维度
                // GLM 使用交错模式：对 (even, odd) 索引对应用 RoPE
                int even_idx = tid * 2;      // 偶数索引：0, 2, 4, ..., 62
                int odd_idx = even_idx + 1;  // 奇数索引：1, 3, 5, ..., 63

                // 读取交错的输入值
                __nv_bfloat16 x1 = x_ptr[even_idx];
                __nv_bfloat16 x2 = x_ptr[odd_idx];
                // 读取对应的 cos/sin 值
                __nv_bfloat16 cos = cos_sin_ptr[tid];
                __nv_bfloat16 sin = cos_sin_ptr[tid + 32];

                // 应用 RoPE 变换到交错的位置
                output_ptr[even_idx] = __hadd(__hmul(x1, cos), __hmul(__hneg(x2), sin));
                output_ptr[odd_idx] = __hadd(__hmul(x2, cos), __hmul(x1, sin));
            } else { // 后 32 个线程处理不变的维度 (tid: 32-63)
                // GLM 的后 32 维保持不变，直接拷贝
                output_ptr[tid + 32] = x_ptr[tid + 32];  // 维度 64-95
                output_ptr[tid + 64] = x_ptr[tid + 64];  // 维度 96-127
            }
        }
    }
}



/**
 * @brief GLM 专用优化缓存推送 RoPE 的主机端接口
 * 
 * 专为 GLM 模型设计的 RoPE 实现，具有以下特点：
 * 1. 交错 RoPE 模式：只对前 64 维中的交错位置应用 RoPE
 * 2. 部分维度保持：后续维度保持原值不变
 * 3. 高效缓存推送：直接写入 KV 缓存
 * 4. GLM 特化优化：针对 GLM 架构的特定优化
 * 
 * 注意：此函数专门为 GLM 模型优化，不适用于其他模型架构
 */
void apply_rotary_pos_emb_push_cache_opt_glm(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output_cache,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size)
{
    // GLM 专用内核启动配置
    apply_rotary_pos_emb_kernel_push_cache_opt_glm<<<seq_len, heads * half_dim>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output_cache.data_ptr<at::BFloat16>()),
        cnts.data_ptr<int32_t>(),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        stride_output_b, stride_output_h, stride_output_s,
        offset_output_s_start, offset_output_s_end,
        half_dim, chunk_size
    );
}

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
 * @file rope.cu
 * @brief 旋转位置编码 (RoPE) 的 CUDA 内核实现
 * 
 * 本文件实现了旋转位置编码 (Rotary Position Embedding) 的基础版本，
 * RoPE 是一种相对位置编码方法，通过旋转变换将位置信息编码到
 * query 和 key 向量中，具有以下优势：
 * 
 * 1. 相对位置感知：能够捕捉 token 之间的相对位置关系
 * 2. 长度外推：训练时的序列长度可以外推到更长的序列
 * 3. 计算高效：通过预计算 cos/sin 值避免重复计算
 * 
 * 实现特点：
 * - 使用 BFloat16 精度平衡性能和精度
 * - 支持批量处理和多头注意力
 * - 优化的内存访问模式
 */

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <vector>
#include "functions.h"

/**
 * @brief 旋转位置编码的 CUDA 内核函数
 * 
 * 实现 RoPE 的核心计算逻辑：
 * 对于输入向量 x = [x1, x2, ..., xd]，将其分为两半 [x1, x2] 和 [x3, x4]，
 * 然后应用旋转变换：
 * - output[i] = x[i] * cos[i] - x[i+d/2] * sin[i]
 * - output[i+d/2] = x[i+d/2] * cos[i+d/2] + x[i] * sin[i+d/2]
 * 
 * @param x 输入张量（query 或 key）
 * @param cos 预计算的余弦值
 * @param sin 预计算的正弦值
 * @param position_ids 位置 ID，用于索引对应的 cos/sin 值
 * @param output 输出张量
 * @param batch_size, heads, seq_len, embed_dim 张量维度信息
 * @param stride_* 各维度的步长信息
 * @param half_dim 嵌入维度的一半
 */
__global__ void apply_rotary_pos_emb_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int64_t* __restrict__ position_ids,
    __nv_bfloat16* __restrict__ output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos, int stride_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    // 获取当前线程处理的批次、头、序列位置
    int b_idx = blockIdx.x;  // 批次索引
    int h_idx = blockIdx.y;  // 注意力头索引
    int s_idx = blockIdx.z;  // 序列位置索引
    int tid = threadIdx.x;   // 线程索引，对应嵌入维度

    // 获取当前位置的 position_id，用于索引 cos/sin 值
    int pid = position_ids[b_idx * stride_pid_b + h_idx * stride_pid_h + s_idx * stride_pid_s];
    const __nv_bfloat16* cos_ptr = cos + pid * stride_cos;
    const __nv_bfloat16* sin_ptr = sin + pid * stride_sin;

    // 计算输入和输出的内存偏移量
    int x_offset = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs;
    const __nv_bfloat16* x_ptr = x + x_offset;
    __nv_bfloat16* output_ptr = output + x_offset;

    // 每个线程处理一对元素 (x[i], x[i+half_dim])
    if (tid < half_dim) {
        // 读取输入向量的两个分量
        __nv_bfloat16 x1 = x_ptr[tid];              // 前半部分
        __nv_bfloat16 x2 = x_ptr[tid + half_dim];   // 后半部分
        
        // 读取对应的 cos/sin 值
        __nv_bfloat16 cos1 = cos_ptr[tid];
        __nv_bfloat16 sin1 = sin_ptr[tid];
        __nv_bfloat16 cos2 = cos_ptr[tid + half_dim];
        __nv_bfloat16 sin2 = sin_ptr[tid + half_dim];

        // 应用旋转变换公式
        // output[i] = x1 * cos1 - x2 * sin1
        output_ptr[tid] = __hadd(__hmul(x1, cos1), __hmul(__hneg(x2), sin1));
        // output[i+half_dim] = x2 * cos2 + x1 * sin2
        output_ptr[tid + half_dim] = __hadd(__hmul(x2, cos2), __hmul(x1, sin2));
    }
}

/**
 * @brief 旋转位置编码的主机端接口函数
 * 
 * 这是从 Python 调用的主要接口，负责设置 CUDA 内核的执行配置
 * 并启动 GPU 计算。
 * 
 * 执行配置说明：
 * - 线程块维度：(batch_size, heads, seq_len) - 每个位置一个线程块
 * - 线程维度：half_dim - 每个线程处理一对旋转元素
 * 
 * 这种配置确保了：
 * 1. 高并行度：所有位置同时处理
 * 2. 内存合并：同一 warp 内的线程访问连续内存
 * 3. 负载均衡：每个线程的工作量相同
 */
void apply_rotary_pos_emb(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos, int stride_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim)
{
    // 设置 CUDA 内核执行配置
    // 每个 (batch, head, seq_pos) 组合对应一个线程块
    const dim3 blocks(batch_size, heads, seq_len);
    // 每个线程块内有 half_dim 个线程，每个线程处理一对旋转元素
    const dim3 threads(half_dim);

    // 启动 CUDA 内核
    apply_rotary_pos_emb_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch_size, heads, seq_len, embed_dim,
        stride_xb, stride_xh, stride_xs, stride_xe,
        stride_cos, stride_sin,
        stride_pid_b, stride_pid_h, stride_pid_s,
        half_dim
    );
}

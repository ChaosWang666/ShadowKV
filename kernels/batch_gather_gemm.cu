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
 * @file batch_gather_gemm.cu
 * @brief 批量稀疏注意力的 GEMM 内核实现
 * 
 * 本文件实现了 ShadowKV 稀疏注意力机制的核心计算内核，主要功能包括：
 * 
 * 1. **稀疏 GEMM 计算**：使用 CUTLASS 库实现高效的稀疏矩阵乘法
 * 2. **批量 Gather 操作**：根据稀疏索引收集相关的 key 和 value
 * 3. **RoPE 集成**：在 GEMM 计算中集成旋转位置编码
 * 4. **内存优化**：通过 gather 操作减少内存访问和计算量
 * 
 * 核心思想：
 * - 只计算稀疏预算 (sparse_budget) 内的重要 token 对
 * - 使用 CUTLASS 的 GemmUniversalBatchGatherIndices 实现高效计算
 * - 支持多头注意力的批量处理
 * - 集成 RoPE 避免额外的内核启动开销
 * 
 * 性能优势：
 * - 显著减少计算复杂度：O(n²) → O(n × sparse_budget)
 * - 高效的 Tensor Core 利用率
 * - 减少内存带宽需求
 */

#include <torch/extension.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

#include "gemm_universal_batch_gather_indices.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "functions.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * CUTLASS GEMM 配置部分
 * 
 * 以下配置定义了稀疏注意力 GEMM 计算的数据类型、内存布局、
 * 线程块配置和计算精度等关键参数。
 */

// 数据类型配置：平衡精度和性能
using ElementAccumulator = float;                  // 累加器数据类型：使用 float 保证精度
using ElementComputeEpilogue = ElementAccumulator; // 后处理计算类型
using ElementInputA = cutlass::bfloat16_t;         // 输入矩阵 A (query) 数据类型
using ElementInputB = cutlass::bfloat16_t;         // 输入矩阵 B (key) 数据类型
using ElementOutput = cutlass::bfloat16_t;         // 输出矩阵数据类型

// 内存布局配置：优化内存访问模式
// 注意：布局选择影响内存合并和缓存效率
using LayoutInputA = cutlass::layout::RowMajor;    // Query 矩阵：行主序 (seq_len × embed_dim)
using LayoutInputB = cutlass::layout::ColumnMajor; // Key 矩阵：列主序 (embed_dim × seq_len)
using LayoutOutput = cutlass::layout::RowMajor;    // 输出矩阵：行主序 (seq_len × seq_len)

// 计算单元配置：使用 Tensor Core 加速
using MMAOp = cutlass::arch::OpClassTensorOp;  // 使用 Tensor Core 而非 CUDA Core

// 目标架构：Ampere (A100/RTX 30xx 系列)
using SmArch = cutlass::arch::Sm80;  // SM 8.0 架构，支持 BF16 Tensor Core

// 线程块和 Warp 配置：优化 Tensor Core 利用率
static const int ShapeMMAThreadBlockN = 128;
// 线程块瓦片大小：M=128, N=128, K=32
// 这个配置在稀疏注意力场景下平衡了并行度和资源利用率
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, ShapeMMAThreadBlockN, 32>;

// Warp 瓦片大小：M=64, N=64, K=32
// 每个 Warp 处理的数据块，影响寄存器使用和共享内存访问
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;

// MMA 操作瓦片大小：针对 Ampere 架构优化
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
// 注意：16,8,8 适用于 Turing；16,8,16 适用于 Ampere

// 线程块调度策略：使用默认的身份调度
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// 后处理操作：线性组合 (alpha * A*B + beta * C)
// 在稀疏注意力中，通常 alpha=1, beta=0，即直接输出 A*B 的结果
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // 输出数据类型
    128 / cutlass::sizeof_bits<ElementOutput>::value, // 向量化内存访问的元素数量
                                                      // 对于 BF16，每次访问 8 个元素 (128/16=8)
                                                      // 这也决定了后处理中数学指令的向量宽度
    ElementAccumulator,                               // 累加器数据类型
    ElementComputeEpilogue>;                          // 线性组合函数中 alpha/beta 的数据类型

// 流水线阶段数：平衡延迟隐藏和资源使用
constexpr int NumStages = 5;
// Ampere 架构推荐 4-5 个阶段，可以更好地隐藏内存延迟
// Turing 架构推荐 2 个阶段，受限于共享内存容量

/**
 * CUTLASS GEMM 模板实例化
 * 
 * 使用 GemmUniversalBatchGatherIndices 实现稀疏注意力的核心计算：
 * - 支持批量处理多个注意力头
 * - 使用 Gather 操作实现稀疏索引访问
 * - 集成 RoPE 位置编码计算
 * - 优化内存访问模式
 */
using Gemm = cutlass::gemm::device::GemmUniversalBatchGatherIndices<
    ElementInputA,                      // Query 矩阵数据类型
    LayoutInputA,                       // Query 矩阵内存布局
    ElementInputB,                      // Key 矩阵数据类型
    LayoutInputB,                       // Key 矩阵内存布局
    ElementOutput,                      // 输出矩阵数据类型
    LayoutOutput,                       // 输出矩阵内存布局
    ElementAccumulator,                 // 累加器精度
    MMAOp,                             // 使用 Tensor Core
    SmArch,                            // 目标架构 SM 8.0
    ShapeMMAThreadBlock,               // 线程块瓦片大小
    ShapeMMAWarp,                      // Warp 瓦片大小
    ShapeMMAOp,                        // MMA 操作大小
    EpilogueOp,                        // 后处理操作
    SwizzleThreadBlock,                // 线程块调度
    NumStages,                         // 流水线阶段数
    8, /*alignmentA*/                  // A 矩阵内存对齐 (8 字节)
    8, /*alignmentB*/                  // B 矩阵内存对齐 (8 字节)
    cutlass::arch::OpMultiplyAdd,      // 基础运算类型
    cutlass::ComplexTransform::kNone,  // A 矩阵复数变换 (无)
    cutlass::ComplexTransform::kNone,  // B 矩阵复数变换 (无)
    true,  /*GatherA*/                 // 启用 Query 矩阵 Gather 操作
    false, /*GatherB*/                 // 禁用 Key 矩阵 Gather 操作
    false  /*ScatterD*/                // 禁用输出矩阵 Scatter 操作
    >;

/**
 * 批量稀疏注意力 GEMM 主函数
 * 
 * 实现 ShadowKV 稀疏注意力的核心计算，结合了 GEMM、Gather 和 RoPE 操作。
 * 
 * @param a Query 矩阵 (batch_size × heads × seq_len × embed_dim)
 * @param b Key 矩阵 (batch_size × heads × embed_dim × seq_len)
 * @param cos RoPE 余弦缓存 (max_seq_len × embed_dim)
 * @param sin RoPE 正弦缓存 (max_seq_len × embed_dim)
 * @param position_ids 稀疏位置索引数组
 * @param output 输出矩阵 (batch_size × heads × sparse_budget × embed_dim)
 * @param batch_size 批量大小
 * @param heads 注意力头数
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param rank Query 矩阵的有效维度
 * @param sparse_budget 稀疏预算 (每个序列保留的 token 数)
 * @param max_seq_len 最大序列长度
 * @param chunk_size 分块大小
 * @param offset_array 批量偏移数组
 */
void batch_gather_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor position_ids,
    torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim, int rank, int sparse_budget,
    int max_seq_len, int chunk_size, torch::Tensor offset_array)
{

    // 初始化线性组合参数
    // alpha: 缩放因子 (设为 1.0，直接输出 GEMM 结果)
    // beta: 偏置项系数 (设为 0.0，不累加到现有结果)
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // 定义 GEMM 问题的原始尺寸
    // M: 序列长度, N: 嵌入维度, K: Query 矩阵的有效维度 (rank)
    cutlass::gemm::GemmCoord problem_size = {seq_len, embed_dim, rank};

    // 定义稀疏 GEMM 的实际计算尺寸
    // M: 稀疏预算 (只计算重要的 token), N: 嵌入维度, K: rank
    // 这是性能优化的关键：从 O(seq_len²) 降低到 O(seq_len × sparse_budget)
    cutlass::gemm::GemmCoord problem_size_real(sparse_budget,
                                               problem_size.n(),
                                               problem_size.k());

    // 创建 CUTLASS GEMM 内核参数结构
    // 这些参数定义了稀疏注意力计算的完整配置，包括：
    // - 矩阵尺寸和批量信息
    // - 内存指针和步长
    // - 稀疏索引和 RoPE 缓存
    // - 批量处理的偏移信息
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kBatched,                          // 批量计算模式
        problem_size_real,                                                   // 稀疏 GEMM 问题尺寸
        batch_size * heads,                                                  // 总批量大小 (批量 × 注意力头数)
        {alpha, beta},                                                       // 线性组合系数
        reinterpret_cast<ElementInputA *>(a.data_ptr<at::BFloat16>()),      // Query 矩阵指针
        reinterpret_cast<ElementInputB *>(b.data_ptr<at::BFloat16>()),      // Key 矩阵指针
        reinterpret_cast<ElementOutput *>(output.data_ptr<at::BFloat16>()), // 输入 C 矩阵指针
        reinterpret_cast<ElementOutput *>(output.data_ptr<at::BFloat16>()), // 输出 D 矩阵指针
        reinterpret_cast<ElementOutput *>(sin.data_ptr<at::BFloat16>()),    // RoPE 正弦缓存指针
        reinterpret_cast<ElementOutput *>(cos.data_ptr<at::BFloat16>()),    // RoPE 余弦缓存指针
        problem_size.m() * problem_size.k(),                                // A 矩阵批量步长
        problem_size.n() * problem_size.k(),                                // B 矩阵批量步长
        sparse_budget * problem_size.n(),                                   // C 矩阵批量步长
        sparse_budget * problem_size.n(),                                   // D 矩阵批量步长
        LayoutInputA::packed(problem_size.mk()).stride(),                   // A 矩阵 leading dimension
        LayoutInputB::packed(problem_size.kn()).stride(),                   // B 矩阵 leading dimension
        LayoutOutput::packed({sparse_budget, problem_size.n()}).stride(),   // C 矩阵 leading dimension
        LayoutOutput::packed({sparse_budget, problem_size.n()}).stride(),   // D 矩阵 leading dimension
        LayoutOutput::packed({max_seq_len, problem_size.n()}).stride(),     // 正弦缓存 leading dimension
        LayoutOutput::packed({max_seq_len, problem_size.n()}).stride(),     // 余弦缓存 leading dimension
        reinterpret_cast<int *>(position_ids.data_ptr<int>()),              // Query 矩阵稀疏索引
        nullptr,                                                             // Key 矩阵 Gather 索引 (未使用)
        nullptr,                                                             // 输出矩阵 Scatter 索引 (未使用)
        reinterpret_cast<int *>(position_ids.data_ptr<int>()),              // 正弦缓存 Gather 索引
        reinterpret_cast<int *>(position_ids.data_ptr<int>()),              // 余弦缓存 Gather 索引
        sparse_budget / chunk_size,                                         // A 索引批量步长
        0,                                                                   // B 索引批量步长
        0,                                                                   // D 索引批量步长
        sparse_budget / chunk_size,                                         // 正弦缓存索引批量步长
        sparse_budget / chunk_size,                                         // 余弦缓存索引批量步长
        max_seq_len,                                                         // 最大序列长度
        chunk_size,                                                          // 分块大小
        heads,                                                               // 注意力头数
        reinterpret_cast<int *>(offset_array.data_ptr<int>())               // 批量偏移数组指针
        };

    // 查询 GEMM 计算所需的额外工作空间大小
    // 某些 CUTLASS 内核需要临时存储空间来优化计算性能
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // 在 GPU 设备上分配工作空间内存
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 实例化 CUTLASS GEMM 操作对象
    // 根据模板参数创建具体的内核实例
    Gemm gemm_op;

    // 验证当前问题尺寸和参数配置是否被支持
    // 确保硬件和软件兼容性
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // 使用参数和工作空间指针初始化 CUTLASS 内核
    // 准备内核的执行环境和资源
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // 启动稀疏注意力 GEMM 计算
    // 执行 Query × Key^T 的稀疏矩阵乘法，同时集成 RoPE 位置编码
    status = gemm_op();
    CUTLASS_CHECK(status);
}

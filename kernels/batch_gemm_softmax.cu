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
 * @file batch_gemm_softmax.cu
 * @brief 批量 GEMM+Softmax 融合内核实现
 * 
 * 本文件实现了 ShadowKV 稀疏注意力机制中的注意力权重计算和归一化，主要功能包括：
 * 
 * 1. **融合 GEMM+Softmax**：将矩阵乘法和 Softmax 操作融合在一个内核中
 * 2. **批量处理**：支持多头注意力的并行计算
 * 3. **内存优化**：减少中间结果的内存读写
 * 4. **数值稳定性**：使用高精度累加器确保 Softmax 计算的稳定性
 * 
 * 核心计算流程：
 * 1. 执行 Query × Key^T 矩阵乘法 (已由 batch_gather_gemm 完成)
 * 2. 计算每行的最大值 (用于数值稳定的 Softmax)
 * 3. 计算 exp(x - max) 和行和
 * 4. 执行 Softmax 归一化：softmax(x) = exp(x - max) / sum
 * 
 * 性能优势：
 * - 避免中间结果的全局内存访问
 * - 利用 Tensor Core 加速 GEMM 计算
 * - 优化的 Softmax 实现，减少数值误差
 * - 支持大规模批量处理
 */

#include <torch/extension.h>

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm.h"
#include "helper.h"

#include "functions.h"

////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"

#include "batch_gemm_softmax.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 数据类型配置
 * 
 * 定义 GEMM 和 Softmax 操作中使用的数据类型，平衡精度和性能。
 */

// GEMM 计算相关数据类型
using ElementA = cutlass::bfloat16_t;        // 输入矩阵 A (Query) 数据类型
using ElementB = cutlass::bfloat16_t;        // 输入矩阵 B (Key) 数据类型
using ElementC = cutlass::bfloat16_t;        // 输入矩阵 C 数据类型
using ElementCompute = float;                // GEMM 计算精度 (使用 float 保证精度)
using ElementD = ElementC;                   // 输出矩阵 D 数据类型 (bfloat16)

// Softmax 计算相关数据类型
using ElementSoftmax = ElementC;             // Softmax 输出数据类型 (bfloat16)
using ElementSoftmaxCompute = float;         // Softmax 内部计算精度 (float)
using ElementNorm = float;                   // 归一化因子数据类型
using ElementSum = float;                    // 行和数据类型

// 内存布局配置
using LayoutA = cutlass::layout::RowMajor;   // Query 矩阵：行主序
using LayoutB = cutlass::layout::ColumnMajor; // Key 矩阵：列主序

// 内存对齐配置：优化内存访问效率
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;     // A 矩阵对齐 (8 元素)
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;     // B 矩阵对齐 (8 元素)
static constexpr int AlignmentSoftmax =
    128 / cutlass::sizeof_bits<ElementSoftmax>::value;                             // Softmax 对齐 (8 元素)

/**
 * 线程块和 Warp 配置
 * 
 * 针对注意力权重计算优化的瓦片大小配置：
 * - ThreadblockShape: 线程块处理的数据瓦片大小
 * - WarpShape: 每个 Warp 处理的数据瓦片大小
 * - InstructionShape: Tensor Core 指令的瓦片大小
 */
using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 32>;  // 线程块瓦片：M=32, N=256, K=32
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;          // Warp 瓦片：M=32, N=64, K=32
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // Tensor Core 指令瓦片

// 计算架构配置
using OperatorClass = cutlass::arch::OpClassTensorOp;            // 使用 Tensor Core
using ArchTag = cutlass::arch::Sm80;                            // 目标架构 SM 8.0

// Softmax 应用形状：定义 Softmax 操作的处理单元
// 1×1024 表示每次处理 1 行，最多 1024 个元素
using ApplyShape = cutlass::MatrixShape<1, 1024>;

// 流水线阶段数：平衡延迟隐藏和资源使用
static int const kStages = 4;

/**
 * 后处理操作配置
 * 
 * 定义 GEMM 结果的线性变换操作：
 * - 支持 alpha 缩放 (通常用于温度缩放)
 * - 为后续 Softmax 计算准备数据
 */
using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                                        // 输出数据类型
    128 / cutlass::sizeof_bits<ElementC>::value,     // 向量化访问宽度
    ElementCompute,                                  // 计算数据类型
    ElementCompute,                                  // 缩放因子数据类型
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling  // 只使用 alpha 缩放
>;

/**
 * 批量 GEMM+Softmax 融合操作模板实例化
 * 
 * 这是 ShadowKV 注意力权重计算的核心组件，将以下操作融合在一个内核中：
 * 1. 批量 GEMM 计算：Query × Key^T
 * 2. 数值稳定的 Softmax 归一化
 * 3. 中间统计量计算 (最大值、行和)
 * 
 * 融合的优势：
 * - 减少全局内存访问
 * - 提高缓存利用率
 * - 降低内核启动开销
 * - 保证数值稳定性
 */
using BatchGemmSoftmax = cutlass::BatchGemmSoftmax<
    ElementA, LayoutA,              // Query 矩阵配置
    ElementB, LayoutB,              // Key 矩阵配置
    ElementC,                       // 输入 C 矩阵数据类型
    ElementCompute,                 // GEMM 计算精度
    OperatorClass,                  // 使用 Tensor Core
    ArchTag,                        // 目标架构
    ThreadblockShape,               // 线程块瓦片大小
    WarpShape,                      // Warp 瓦片大小
    InstructionShape,               // 指令瓦片大小
    EpilogueFunctorOp,              // 后处理操作
    kStages,                        // 流水线阶段数
    ApplyShape,                     // Softmax 应用形状
    AlignmentA,                     // A 矩阵内存对齐
    AlignmentB,                     // B 矩阵内存对齐
    AlignmentSoftmax,               // Softmax 内存对齐
    ElementNorm,                    // 归一化因子数据类型
    ElementSum,                     // 行和数据类型
    ElementSoftmax,                 // Softmax 输出数据类型
    ElementSoftmaxCompute           // Softmax 计算精度
>;

// 从模板实例中提取布局类型
using LayoutC = typename BatchGemmSoftmax::LayoutC;     // 输出矩阵布局
using LayoutN = typename BatchGemmSoftmax::LayoutN;     // 归一化因子布局
using LayoutS = typename BatchGemmSoftmax::LayoutS;     // Softmax 输出布局
using MatrixCoord = typename LayoutC::TensorCoord;      // 矩阵坐标类型

/**
 * 批量 GEMM+Softmax 融合计算主函数
 * 
 * 执行稀疏注意力的权重计算和归一化，包括：
 * 1. Query × Key^T 矩阵乘法
 * 2. 温度缩放 (通过 alpha 参数)
 * 3. 数值稳定的 Softmax 计算
 * 4. 统计量输出 (最大值、行和)
 * 
 * @param A Query 矩阵 (batch_count × m × k)
 * @param B Key 矩阵 (batch_count × k × n)
 * @param D 输出注意力权重矩阵 (batch_count × m × n)
 * @param Norm 归一化因子输出 (batch_count × m)
 * @param Sum 行和输出 (batch_count × m)
 * @param Softmax Softmax 结果输出 (batch_count × m × n)
 * @param batch_count 批量大小
 * @param m 序列长度 (Query 序列长度)
 * @param n 键值序列长度 (Key 序列长度)
 * @param k 嵌入维度
 * @param alpha 温度缩放因子 (默认 1.0)
 * @param beta 偏置项系数 (默认 0.0)
 */
void batch_gemm_softmax(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha = 1.0,
    float beta = 0.0
) {
    // 定义 GEMM 问题尺寸
    cutlass::gemm::GemmCoord problem = {m, n, k};

    // 计算各矩阵的 leading dimension (内存步长)
    int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);  // Query 矩阵步长
    int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);  // Key 矩阵步长
    int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);  // 输出矩阵步长

    // 归一化因子和行和使用固定的行主序布局
    int64_t ldn = problem.m();  // 归一化因子步长
    int64_t lds = problem.m();  // 行和步长

    // 计算所需的线程块数量
    // 基于 N 维度和线程块的 N 瓦片大小计算
    int block_num = (problem.n() + BatchGemmSoftmax::ThreadblockShape::kN - 1) / BatchGemmSoftmax::ThreadblockShape::kN;

    // 创建批量 GEMM+Softmax 操作的参数结构
    BatchGemmSoftmax::Arguments args(
      problem,                                                    // GEMM 问题尺寸
      batch_count,                                               // 批量大小
      {reinterpret_cast<ElementA *>(A.data_ptr()), lda},        // Query 矩阵指针和步长
      {reinterpret_cast<ElementB *>(B.data_ptr()), ldb},        // Key 矩阵指针和步长
      {nullptr, ldc},                                            // 输入 C 矩阵 (通常为 nullptr)
      {reinterpret_cast<ElementD *>(D.data_ptr()), ldc},        // 输出 D 矩阵指针和步长
      {
        ElementCompute(alpha),                                   // 温度缩放因子
        ElementCompute(beta)                                     // 偏置项系数
      },
      {reinterpret_cast<ElementNorm *>(Norm.data_ptr()), ldn},  // 归一化因子输出
      {reinterpret_cast<ElementSum *>(Sum.data_ptr()), lds},    // 行和输出
      {reinterpret_cast<ElementSoftmax *>(Softmax.data_ptr()), ldc}, // Softmax 结果输出
      problem.m() * problem.k(),                                 // A 矩阵批量步长
      problem.k() * problem.n(),                                 // B 矩阵批量步长
      problem.m() * problem.n(),                                 // C 矩阵批量步长
      problem.m() * problem.n(),                                 // D 矩阵批量步长
      block_num * problem.m(),                                   // 归一化因子批量步长
      block_num * problem.m(),                                   // 行和批量步长
      problem.m() * problem.n()                                  // Softmax 批量步长
    );

    // 实例化批量 GEMM+Softmax 操作对象
    BatchGemmSoftmax batch_gemm_softmax;

    // 使用参数初始化操作对象
    // 验证参数有效性并准备执行环境
    CUTLASS_CHECK(batch_gemm_softmax.initialize(args));

    // 执行融合的 GEMM+Softmax 计算
    // 包括矩阵乘法、温度缩放和 Softmax 归一化
    CUTLASS_CHECK(batch_gemm_softmax());
}
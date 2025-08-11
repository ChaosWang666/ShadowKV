/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
 * @file batch_gemm_softmax.h
 * @brief ShadowKV 批量 GEMM+Softmax 融合操作头文件
 * 
 * 本文件定义了 ShadowKV 稀疏注意力机制中的核心计算组件，实现了高效的
 * 批量矩阵乘法和 Softmax 归一化融合操作。主要包含：
 * 
 * 1. **ApplySoftmax 内核**：执行数值稳定的 Softmax 计算
 * 2. **BatchGemmSoftmax 类**：融合 GEMM 和 Softmax 的完整操作
 * 3. **模板化设计**：支持多种数据类型和计算精度配置
 * 4. **CUTLASS 集成**：利用 NVIDIA CUTLASS 库的高性能 GEMM 实现
 * 
 * 核心优势：
 * - 减少内存带宽需求：避免中间结果的全局内存读写
 * - 数值稳定性：使用 max-subtraction 技术防止 Softmax 溢出
 * - 高性能计算：利用 Tensor Core 加速矩阵运算
 * - 批量处理：支持多头注意力的并行计算
 * 
 * 适用场景：
 * - Transformer 模型的注意力权重计算
 * - ShadowKV 稀疏注意力机制
 * - 大规模语言模型推理优化
 */

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"

#include "cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/reduction/kernel/reduce_softmax_final.h"

#include "batch_gemm_with_epilogue_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @class ApplySoftmax
 * @brief 数值稳定的 Softmax 应用内核
 * 
 * 本类实现了高效且数值稳定的 Softmax 计算，是 ShadowKV 注意力权重归一化的核心组件。
 * 采用经典的 max-subtraction 技术确保计算的数值稳定性，避免指数函数溢出。
 * 
 * 计算流程：
 * 1. 读取 GEMM 输出矩阵 D (Query × Key^T 结果)
 * 2. 读取预计算的归一化因子 N (每行最大值)
 * 3. 读取预计算的行和 S (exp(x - max) 的行和)
 * 4. 计算最终 Softmax：softmax(x) = exp(x - max) / sum
 * 
 * 模板参数说明：
 * @tparam ElementD_ GEMM 输出矩阵的数据类型 (通常为 bfloat16)
 * @tparam ElementNorm_ 归一化因子的数据类型 (通常为 float)
 * @tparam ElementSum_ 行和的数据类型 (通常为 float)
 * @tparam ElementSoft_ Softmax 输出的数据类型 (通常为 bfloat16)
 * @tparam ElementSoftmaxCompute_ Softmax 内部计算精度 (通常为 float)
 * @tparam Alignment 内存对齐要求 (优化内存访问性能)
 * @tparam ApplyShape_ 处理形状 (默认 1×1024，表示每次处理 1 行最多 1024 个元素)
 */
template <typename ElementD_, typename ElementNorm_, typename ElementSum_,
          typename ElementSoft_, typename ElementSoftmaxCompute_, int Alignment,
          typename ApplyShape_ = MatrixShape<1, 1024>>
class ApplySoftmax {
public:
  using ElementD = ElementD_;
  using ElementNorm = ElementNorm_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoft_;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;

  static int const kAlignment = Alignment;
  using ApplyShape = ApplyShape_;

  using Layout = cutlass::layout::RowMajor;

  using TensorRefD = TensorRef<ElementD, Layout>;
  using TensorRefN = TensorRef<ElementNorm, Layout>;
  using TensorRefSum = TensorRef<ElementSum, Layout>;
  using TensorRefSoft = TensorRef<ElementSoft, Layout>;

  using FragmentSoftmax = Array<ElementSoftmaxCompute, kAlignment>;

  /**
   * @struct Arguments
   * @brief ApplySoftmax 内核的参数配置结构
   * 
   * 包含执行 Softmax 计算所需的所有输入输出张量引用和批量处理配置。
   * 支持批量处理多个注意力头的并行计算。
   */
  struct Arguments {

    MatrixCoord extent;        ///< D 和 Softmax 矩阵的尺寸 (m × n)
    int batch_count;           ///< 批量大小 (注意力头数量)
    TensorRefD ref_D;          ///< GEMM+Max 计算结果矩阵 D (输入)
    TensorRefN ref_N;          ///< 归一化因子张量 N (输入，每行最大值)
    TensorRefSum ref_S;        ///< 行和张量 S (输入，exp(x-max) 的行和)
    TensorRefSoft ref_Soft;    ///< Softmax 输出张量 (输出)
    int64_t batch_stride_D;    ///< D 张量的批量步长
    int64_t batch_stride_N;    ///< N 张量的批量步长
    int64_t batch_stride_S;    ///< S 张量的批量步长
    int64_t batch_stride_Soft; ///< Softmax 张量的批量步长

    //
    // Methods
    //
    Arguments()
        : batch_count(1), batch_stride_D(0), batch_stride_N(0),
          batch_stride_S(0), batch_stride_Soft(0) {}

    Arguments(MatrixCoord extent_, ///< Extent of D and Softmax matrices
              int batch_count_,    ///< Batch count
              TensorRefD ref_D_,   ///< D matrix computed by GEMM+PartialReduce
              TensorRefN ref_N_,   ///< Output parameter for N
              TensorRefSum ref_S_, ///< Output parameter for N
              TensorRefSoft ref_Soft_, ///< Softmax
              int64_t batch_stride_D_ = 0, int64_t batch_stride_N_ = 0,
              int64_t batch_stride_S_ = 0, int64_t batch_stride_Soft_ = 0)
        : extent(extent_), batch_count(batch_count_), ref_D(ref_D_),
          ref_N(ref_N_), ref_S(ref_S_), ref_Soft(ref_Soft_),
          batch_stride_D(batch_stride_D_), batch_stride_N(batch_stride_N_),
          batch_stride_S(batch_stride_S_),
          batch_stride_Soft(batch_stride_Soft_) {}
  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() {}

    Params(Arguments const &args_) : args(args_) {}
  };

  //
  // SharedStorage
  //

  struct SharedStorage {};

private:
public:
  CUTLASS_DEVICE
  ApplySoftmax() {}

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    apply(params, shared_storage);
  }

private:
  /// Compute Softmax
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * ApplyShape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    int batch_offset_norm = block_batch * params.args.batch_stride_N;
    int batch_offset_sum = block_batch * params.args.batch_stride_S;

    // Kill off thread if it is outside the row boundary
    if (params.args.extent.row() <= idx_m) {
      return;
    }

    //
    // Setup pointers to load D again
    //

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;
    using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;
    using FragmentSoft = Array<ElementSoft, kAlignment>;
    using FragmentSoftAligned = AlignedArray<ElementSoft, kAlignment>;

    using ConvertSoftCompute =
        cutlass::NumericArrayConverter<ElementSoftmaxCompute, ElementD,
                                       kAlignment>;
    using ConvertSoftOutput =
        cutlass::NumericArrayConverter<ElementSoft, ElementSoftmaxCompute,
                                       kAlignment>;

    using Mul = cutlass::multiplies<FragmentSoftmax>;
    using Minus = cutlass::minus<FragmentSoftmax>;
    using Exp = cutlass::fast_exp_op<FragmentSoftmax>;

    ConvertSoftCompute convert_soft_compute;
    ConvertSoftOutput convert_soft_output;

    Minus minus;
    Mul mul;
    Exp exponential;

    using ConvertSum =
        cutlass::NumericConverter<ElementSoftmaxCompute, ElementSum>;
    using ConvertNorm =
        cutlass::NumericConverter<ElementSoftmaxCompute, ElementNorm>;

    ConvertSum convert_sum;
    ConvertNorm convert_norm;

    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
        params.args.ref_D.data() + params.args.batch_stride_D * block_batch +
        params.args.ref_D.layout()({idx_m, idx_n}));

    AccessTypeSoft *access_soft = reinterpret_cast<AccessTypeSoft *>(
        params.args.ref_Soft.data() +
        params.args.batch_stride_Soft * block_batch +
        params.args.ref_Soft.layout()({idx_m, idx_n}));

    ElementSum inv_sum = (params.args.ref_S.data())[idx_m + batch_offset_sum];
    ElementNorm norm = (params.args.ref_N.data())[idx_m + batch_offset_norm];

    //
    // Loop
    //
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < params.args.extent.column();
         idx += ApplyShape::kColumn * kAlignment) {

      if (idx_n < params.args.extent.column()) {
        AccessTypeD fetch;
        arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d,
                                                            true);

        FragmentSoftmax result = mul(
            exponential(minus(convert_soft_compute(fetch), convert_norm(norm))),
            convert_sum(inv_sum));
        FragmentSoft soft = convert_soft_output(result);

        arch::global_store<FragmentSoft, sizeof(FragmentSoft)>(
            soft, access_soft, true);
      }

      access_d += ApplyShape::kColumn;
      access_soft += ApplyShape::kColumn;
      idx_n += ApplyShape::kColumn * kAlignment;
    }
  }
};

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @class BatchGemmSoftmax
 * @brief ShadowKV 批量 GEMM+Softmax 融合操作的主类
 * 
 * 本类是 ShadowKV 稀疏注意力机制的核心计算组件，实现了高效的批量矩阵乘法
 * 和 Softmax 归一化融合操作。通过将多个计算步骤融合在一起，显著减少了
 * 内存访问开销并提高了计算效率。
 * 
 * 主要功能：
 * 1. **批量 GEMM 计算**：执行 Query × Key^T 矩阵乘法
 * 2. **温度缩放**：应用注意力温度参数进行缩放
 * 3. **数值稳定 Softmax**：计算注意力权重的归一化
 * 4. **统计量输出**：提供最大值和行和用于后续计算
 * 
 * 性能优化特性：
 * - 利用 CUTLASS 库的高性能 GEMM 实现
 * - 使用 Tensor Core 加速矩阵运算
 * - 融合计算减少内存带宽需求
 * - 支持多种数据类型和精度配置
 * - 优化的线程块和 Warp 配置
 * 
 * 模板参数说明：
 * @tparam ElementA_ Query 矩阵数据类型
 * @tparam LayoutA_ Query 矩阵内存布局
 * @tparam ElementB_ Key 矩阵数据类型
 * @tparam LayoutB_ Key 矩阵内存布局
 * @tparam ElementC_ 输出矩阵数据类型
 * @tparam ElementCompute_ GEMM 计算精度
 * @tparam OperatorClass_ 计算类型 (TensorOp/SIMT)
 * @tparam ArchTag_ 目标 GPU 架构
 * @tparam ThreadblockShape_ 线程块瓦片形状
 * @tparam WarpShape_ Warp 瓦片形状
 * @tparam InstructionShape_ 指令瓦片形状
 * @tparam EpilogueFunctorOp_ Epilogue 函数操作
 * @tparam kStages_ 流水线阶段数
 * @tparam ApplyShape_ Softmax 应用形状
 * @tparam AlignmentA_ Query 矩阵内存对齐
 * @tparam AlignmentB_ Key 矩阵内存对齐
 * @tparam AlignmentSoftmax_ Softmax 内存对齐
 * @tparam ElementNorm_ 归一化因子数据类型
 * @tparam ElementSum_ 行和数据类型
 * @tparam ElementSoftmax_ Softmax 输出数据类型
 * @tparam ElementSoftmaxCompute_ Softmax 计算精度
 */
template <typename ElementA_, typename LayoutA_, typename ElementB_,
          typename LayoutB_, typename ElementC_, typename ElementCompute_,
          typename OperatorClass_, typename ArchTag_,
          typename ThreadblockShape_, typename WarpShape_,
          typename InstructionShape_, typename EpilogueFunctorOp_, int kStages_,
          typename ApplyShape_ = MatrixShape<1, 1024>,
          int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
          int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value,
          int AlignmentSoftmax_ = 128 / cutlass::sizeof_bits<ElementC_>::value,
          typename ElementNorm_ = float, typename ElementSum_ = float,
          typename ElementSoftmax_ = ElementC_,
          typename ElementSoftmaxCompute_ = ElementCompute_>
class BatchGemmSoftmax {
public:
  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoftmax_;
  // using ElementSoftmaxCompute = float;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;
  using ElementNorm = ElementNorm_;

  using ApplyShape = ApplyShape_;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutS = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementNorm, LayoutN>;
  using TensorRefSum = TensorRef<ElementSum, LayoutS>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;

  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  static int const kStages = kStages_;
  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;
  static int const AlignmentSoftmax = AlignmentSoftmax_;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementC,
      LayoutC, ElementCompute, OperatorClass, ArchTag, ThreadblockShape,
      WarpShape, InstructionShape, EpilogueFunctorOp, ThreadblockSwizzle,
      kStages, true,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OperatorClass, ArchTag, ElementA, ElementB, ElementC,
          ElementCompute>::Operator,
      cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

  // Epilogue visitor
  using EpilogueVisitor =
      typename cutlass::epilogue::threadblock::EpilogueVisitorSoftmax<
          ThreadblockShape, DefaultGemmKernel::kThreadCount,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementCompute, ElementNorm, ElementSum, ElementSoftmaxCompute,
          EpilogueFunctorOp>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor, typename DefaultGemmKernel::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel =
      gemm::kernel::BatchGemmWithEpilogueVisitor<typename DefaultGemmKernel::Mma,
                                            Epilogue, ThreadblockSwizzle>;

  // Softmax kernel
  using SoftmaxApplyKernel =
      kernel::ApplySoftmax<ElementC, ElementNorm, ElementSum, ElementSoft,
                           ElementSoftmaxCompute, AlignmentSoftmax, ApplyShape>;

  using ApplyFinalReductionKernel =
      cutlass::reduction::kernel::ApplySoftmaxFinalReduction<
          ElementNorm, ElementSum, ElementSoftmaxCompute, ThreadblockShape>;

public:
  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments gemm;
    typename SoftmaxApplyKernel::Arguments softmax;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() {}

    Arguments(cutlass::gemm::GemmCoord problem_size, int32_t batch_count_,
              TensorRefA ref_A_, TensorRefB ref_B_, TensorRefC ref_C_,
              TensorRefC ref_D_,
              typename EpilogueFunctorOp::Params linear_scaling,
              TensorRefN ref_N_, TensorRefSum ref_S_,
              TensorRefSoft ref_Softmax_, int64_t batch_stride_A_ = 0,
              int64_t batch_stride_B_ = 0, int64_t batch_stride_C_ = 0,
              int64_t batch_stride_D_ = 0, int64_t batch_stride_Max_ = 0,
              int64_t batch_stride_Sum_ = 0, int64_t batch_stride_Softmax_ = 0)
        : gemm(cutlass::gemm::GemmUniversalMode::kBatched, problem_size,
               batch_count_, ref_A_, ref_B_, ref_C_, ref_D_, ref_N_.data(),
               ref_S_.data(), batch_stride_A_, batch_stride_B_,
               typename EpilogueVisitor::Arguments(
                   linear_scaling, batch_stride_C_, batch_stride_D_,
                   batch_stride_Max_, batch_stride_Sum_)),
          reduction(problem_size, ref_N_.data(), ref_S_.data(),
                    batch_stride_Max_, batch_stride_Sum_),
          softmax(MatrixCoord(problem_size.m(), problem_size.n()), batch_count_,
                  ref_D_, ref_N_, ref_S_, ref_Softmax_, batch_stride_D_,
                  batch_stride_Max_, batch_stride_Sum_, batch_stride_Softmax_),
          extend(problem_size) {}
  };

  struct Params {

    typename GemmKernel::Params gemm;
    typename SoftmaxApplyKernel::Params softmax;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() {}

    Params(Arguments const &args)
        : gemm(args.gemm), reduction(args.reduction), softmax(args.softmax),
          extend(MatrixCoord(args.extend.m(), args.extend.n())) {}
  };

private:
  Params params_;

public:
  /// Ctor
  BatchGemmSoftmax() {}

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the BatchGEMM + max kernel
    //

    dim3 gemm_grid =
        ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel>
        <<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the ApplyFinalReductionKernel
    //

    int thread_per_block = 128;
    int block_per_row =
        (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row =
          (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_grid(block_per_row, 1,
                              params_.softmax.args.batch_count);
    dim3 final_reduction_block(thread_per_block);

    Kernel<ApplyFinalReductionKernel>
        <<<final_reduction_grid, final_reduction_block,
           sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream>>>(
            params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(SoftmaxApplyKernel::ApplyShape::kColumn,
                     SoftmaxApplyKernel::ApplyShape::kRow);

    int threadblock_rows = SoftmaxApplyKernel::ApplyShape::kRow;
    int threadblock_columns = SoftmaxApplyKernel::ApplyShape::kColumn *
                              SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
        (params_.softmax.args.extent.row() + threadblock_rows - 1) /
            threadblock_rows,
        (params_.softmax.args.extent.column() + threadblock_columns - 1) /
            threadblock_columns,
        params_.softmax.args.batch_count);

    Kernel<SoftmaxApplyKernel>
        <<<apply_grid, apply_block,
           sizeof(typename SoftmaxApplyKernel::SharedStorage), stream>>>(
            params_.softmax);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }
};

} // namespace cutlass
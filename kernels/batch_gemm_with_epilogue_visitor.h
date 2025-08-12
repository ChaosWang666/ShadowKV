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
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief GEMM kernel to support the epilogue visitor model
    for customized softmax partial reduction epilogue fusion.
    
    文件功能：支持 epilogue visitor 模型的 GEMM 内核
    主要用途：为 ShadowKV 的稀疏注意力机制提供自定义 softmax 部分归约 epilogue 融合
    
    核心特性：
    1. Epilogue Visitor 模式：支持自定义的 epilogue 操作访问者模式
    2. Softmax 融合：将 softmax 计算直接融合到 GEMM epilogue 中
    3. 部分归约：支持分块的 softmax 归约操作，提高数值稳定性
    4. 批处理支持：支持批量 GEMM 操作，适用于多头注意力
    5. 内存优化：通过 epilogue 融合减少中间结果的内存访问
    
    性能优势：
    - 减少内存带宽：避免存储中间的注意力权重矩阵
    - 提高数值稳定性：使用在线 softmax 算法
    - 降低延迟：GEMM 和 softmax 操作流水线化
    - 支持稀疏模式：配合 ShadowKV 的稀疏注意力策略
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/**
 * BatchGemmWithEpilogueVisitor - 支持 Epilogue Visitor 的批量 GEMM 内核
 * 
 * 这是 ShadowKV 稀疏注意力机制的核心计算内核，将 GEMM 计算与自定义的
 * epilogue 操作（如 softmax）融合在一起，实现高效的注意力权重计算。
 * 
 * 模板参数：
 * @param Mma_ 线程块级别的矩阵乘累加操作器
 * @param Epilogue_ Epilogue 操作器，负责后处理和输出
 * @param ThreadblockSwizzle_ 线程块调度函数，优化 GPU 资源利用
 * 
 * 设计理念：
 * - 融合计算：将 GEMM 和 softmax 操作融合，减少内存访问
 * - 访问者模式：通过 EpilogueVisitor 支持灵活的后处理操作
 * - 批处理优化：支持多个 GEMM 操作的批量执行
 * - 数值稳定性：集成在线 softmax 算法，避免数值溢出
 */
template <typename Mma_,      ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_, ///! Epilogue
          typename ThreadblockSwizzle_ ///! Threadblock swizzling function
          >
struct BatchGemmWithEpilogueVisitor {
public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  using ElementNorm = typename EpilogueVisitor::ElementNorm;
  using ElementSum = typename EpilogueVisitor::ElementSum;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  //
  // Structures
  //

  /// Argument structure
  /**
   * Arguments - GEMM 操作的参数结构
   * 
   * 包含执行批量 GEMM 操作所需的所有参数，包括输入矩阵、输出矩阵、
   * 以及 softmax 计算所需的辅助数据结构。
   */
  struct Arguments {

    //
    // Data members - 数据成员
    //

    GemmUniversalMode mode;        // GEMM 执行模式（标准/分割K等）
    GemmCoord problem_size;        // 问题规模 (M, N, K)
    int batch_count;               // 批处理数量

    TensorRefA ref_A;              // 输入矩阵 A 的张量引用
    TensorRefB ref_B;              // 输入矩阵 B 的张量引用
    TensorRefC ref_C;              // 输入矩阵 C 的张量引用（bias）
    TensorRefC ref_D;              // 输出矩阵 D 的张量引用

    ElementNorm *ptr_Max;          // softmax 最大值数组指针（数值稳定性）
    ElementSum *ptr_Sum;           // softmax 归一化因子数组指针

    int64_t batch_stride_A;        // 矩阵 A 的批次间步长
    int64_t batch_stride_B;        // 矩阵 B 的批次间步长

    typename EpilogueVisitor::Arguments epilogue_visitor;  // Epilogue 访问者参数

    //
    // Methods
    //

    Arguments() : mode(GemmUniversalMode::kGemm), batch_count(1) {}

    /// constructs an arguments structure
    Arguments(GemmUniversalMode mode_, GemmCoord problem_size_,
              int batch_count_, TensorRefA ref_A_, TensorRefB ref_B_,
              TensorRefC ref_C_, TensorRefC ref_D_, ElementNorm *ptr_Max_,
              ElementSum *ptr_Sum_, int64_t batch_stride_A_,
              int64_t batch_stride_B_,
              typename EpilogueVisitor::Arguments epilogue_visitor_)
        : mode(mode_), problem_size(problem_size_), batch_count(batch_count_),
          ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_),
          ptr_Max(ptr_Max_), ptr_Sum(ptr_Sum_), batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_), epilogue_visitor(epilogue_visitor_) {

    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  /**
   * Params - 内核执行参数结构
   * 
   * 从 Arguments 转换而来的内核执行参数，包含所有运行时需要的
   * 配置信息和数据指针，针对 GPU 内核执行进行了优化。
   */
  struct Params {

    cutlass::gemm::GemmCoord problem_size;    // GEMM 问题规模
    cutlass::gemm::GemmCoord grid_tiled_shape; // 网格瓦片形状
    int swizzle_log_tile;                     // 瓦片调度的对数参数

    typename Mma::IteratorA::Params params_A; // 矩阵 A 迭代器参数
    typename Mma::IteratorB::Params params_B; // 矩阵 B 迭代器参数
    typename EpilogueVisitor::OutputTileIterator::Params params_C; // 输出 C 迭代器参数
    typename EpilogueVisitor::OutputTileIterator::Params params_D; // 输出 D 迭代器参数

    GemmUniversalMode mode;                   // GEMM 执行模式
    int batch_count;                          // 批处理数量
    int gemm_k_size;                          // K 维度大小

    void *ptr_A;                              // 矩阵 A 数据指针
    void *ptr_B;                              // 矩阵 B 数据指针
    ElementC *ptr_C;                          // 矩阵 C 数据指针
    ElementC *ptr_D;                          // 矩阵 D 数据指针

    ElementNorm *ptr_Max;                     // softmax 最大值数组
    ElementSum *ptr_Sum;                      // softmax 求和数组

    int64_t batch_stride_A;                   // 矩阵 A 批次步长
    int64_t batch_stride_B;                   // 矩阵 B 批次步长

    typename EpilogueVisitor::Params epilogue_visitor; // Epilogue 访问者参数

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0), params_A(0), params_B(0), params_C(0),
          params_D(0), batch_count(0), gemm_k_size(0),
          mode(cutlass::gemm::GemmUniversalMode::kGemm), ptr_A(nullptr),
          ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr), ptr_Max(nullptr),
          ptr_Sum(nullptr), batch_stride_A(0), batch_stride_B(0) {}

    Params(Arguments const &args)
        : problem_size(args.problem_size), swizzle_log_tile(0),
          params_A(args.ref_A.layout()), params_B(args.ref_B.layout()),
          params_C(args.ref_C.layout()), params_D(args.ref_D.layout()),
          mode(args.mode), batch_count(args.batch_count),
          gemm_k_size(args.problem_size.k()), ptr_A(args.ref_A.data()),
          ptr_B(args.ref_B.data()), ptr_C(args.ref_C.data()),
          ptr_D(args.ref_D.data()), ptr_Max(args.ptr_Max),
          ptr_Sum(args.ptr_Sum), batch_stride_A(args.batch_stride_A),
          batch_stride_B(args.batch_stride_B),
          epilogue_visitor(args.epilogue_visitor) {

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          args.batch_count);

      if (args.mode == GemmUniversalMode::kGemm ||
          args.mode == GemmUniversalMode::kGemmSplitKParallel) {

        int const kAlignK =
            const_max(const_max(128 / sizeof_bits<ElementA>::value,
                                128 / sizeof_bits<ElementB>::value),
                      1);

        gemm_k_size = round_up(
            ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

        if (gemm_k_size) {
          grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
        }
      }

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  /**
   * SharedStorage - 共享内存存储联合体
   * 
   * 使用联合体来优化共享内存使用，main_loop 和 epilogue 阶段
   * 复用同一块共享内存空间，减少内存需求。
   * 
   * 内存布局策略：
   * - main_loop: MMA 主循环阶段使用的共享内存
   * - epilogue: Epilogue 阶段使用的共享内存
   *   - epilogue: 标准 epilogue 操作的共享内存
   *   - visitor: EpilogueVisitor 特定操作的共享内存
   */
  union SharedStorage {

    typename Mma::SharedStorage main_loop;    // 主循环共享内存

    struct {
      typename Epilogue::SharedStorage epilogue;  // Epilogue 共享内存
      typename EpilogueVisitor::SharedStorage visitor; // Visitor 共享内存
    } epilogue;
  };

public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  BatchGemmWithEpilogueVisitor() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const &problem_size) {

    CUTLASS_TRACE_HOST("BatchGemmWithEpilogueVisitor::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
        Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<32>>::value ||
               platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

#define SPLIT_K_ENABLED 1

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

#if SPLIT_K_ENABLED
    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm ||
        params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    } else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    } else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA *const *>(
          params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB *const *>(
          params.ptr_B)[threadblock_tile_offset.k()];
    }
#endif

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{offset_k, threadblock_tile_offset.n() *
                                                   Mma::Shape::kN};

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
        params.params_A, ptr_A, {params.problem_size.m(), problem_size_k},
        thread_idx, tb_offset_A);

    typename Mma::IteratorB iterator_B(
        params.params_B, ptr_B, {problem_size_k, params.problem_size.n()},
        thread_idx, tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations =
        (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // assume identity swizzle
    MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                   threadblock_tile_offset.n() *
                                       Mma::Shape::kN);

    int block_idx = threadblock_tile_offset.m() +
                    threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    //
    // Construct the epilogue visitor
    //

    EpilogueVisitor epilogue_visitor(
        params.epilogue_visitor, shared_storage.epilogue.visitor,
        params.problem_size.mn(), thread_idx, warp_idx, lane_idx,
        params.params_C, params.params_D, params.ptr_C, params.ptr_D,
        params.ptr_Max, params.ptr_Sum, threadblock_offset,
        blockIdx.y * params.problem_size.m());

    if (params.mode == GemmUniversalMode::kGemm) {
      // Indicate which position in a serial reduction the output operator is
      // currently updating
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(),
                                       params.grid_tiled_shape.k());
    } else if (params.mode == GemmUniversalMode::kBatched ||
               params.mode == GemmUniversalMode::kArray) {
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    // Construct the epilogue
    Epilogue epilogue(shared_storage.epilogue.epilogue, thread_idx, warp_idx,
                      lane_idx);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(epilogue_visitor, accumulators);
  }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
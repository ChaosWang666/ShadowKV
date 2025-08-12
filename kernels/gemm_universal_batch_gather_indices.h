/*
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file
    \brief 支持批量聚集索引的通用GEMM内核 - ShadowKV稀疏注意力机制的核心组件
    
    本文件实现了一个高度优化的GEMM内核，专门为ShadowKV的稀疏注意力机制设计。
    主要功能包括：
    
    核心特性：
    - 稀疏索引支持：通过gather/scatter操作实现稀疏矩阵访问
    - RoPE集成：内置旋转位置编码支持，优化Transformer模型性能
    - 批量处理：高效处理多个GEMM操作，减少内核启动开销
    - 内存优化：智能内存访问模式，最大化带宽利用率
    - 架构适配：针对不同GPU架构（Volta/Turing/Ampere）的专门优化
    
    技术优势：
    - 融合操作：将GEMM、RoPE、稀疏访问融合在单个内核中
    - 高效同步：优化的线程块调度和同步机制
    - 灵活配置：支持多种数据类型和矩阵布局
    - 性能调优：针对稀疏注意力模式的专门优化
    
    应用场景：
    - 稀疏注意力计算：Q*K^T矩阵乘法与稀疏模式结合
    - KV缓存管理：高效的键值缓存更新和访问
    - 位置编码融合：RoPE与注意力计算的一体化处理
    - 批量推理：多序列并行处理优化
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/params_universal_base.h"
#include "cutlass/trace.h"
#include "cutlass/arch/mma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/layout/permute.h"
#include "cutlass/gemm/kernel/gemm_universal_streamk.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

#include "batch_gather_gemm_epilogue.h"
#include "special_batch_gather_predicated_tile_iterator.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief 支持批量聚集索引的通用GEMM内核类
 * 
 * 这是ShadowKV稀疏注意力机制的核心计算内核，专门设计用于处理带有稀疏索引的
 * 批量GEMM操作。该类将矩阵乘法、稀疏访问模式和RoPE位置编码融合在一起，
 * 为Transformer模型提供高效的注意力计算支持。
 * 
 * 主要特性：
 * - 稀疏矩阵访问：支持通过索引数组进行gather/scatter操作
 * - 批量处理：同时处理多个GEMM操作，提高GPU利用率
 * - RoPE集成：内置旋转位置编码计算，减少内存访问
 * - 灵活配置：支持多种数据类型、矩阵布局和计算精度
 * 
 * @tparam Mma_ 线程块级矩阵乘累加器，定义计算核心逻辑
 * @tparam Epilogue_ 后处理器，处理输出变换和写回操作
 * @tparam ThreadblockSwizzle_ 线程块调度器，优化内存访问模式
 */
template <
  typename Mma_,                  ///! 线程块级矩阵乘累加器 - 定义核心计算逻辑和数据流
  typename Epilogue_,             ///! 后处理器 - 处理输出变换、缩放和内存写回
  typename ThreadblockSwizzle_    ///! 线程块调度器 - 优化内存访问和负载均衡
>
class GemmUniversalBatchGatherIndices{
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  //
  // Structures
  //

  /**
   * @brief GEMM操作参数结构
   * 
   * 包含执行批量聚集索引GEMM操作所需的全部参数，支持稀疏注意力计算的
   * 复杂需求。该结构继承自UniversalArgumentsBase，扩展了稀疏索引、
   * RoPE缓存和批量处理的专门支持。
   * 
   * 核心功能：
   * - 矩阵数据管理：A、B、C、D矩阵的指针和步长信息
   * - 稀疏索引支持：gather/scatter操作的索引数组
   * - RoPE缓存：sin/cos查找表的高效访问
   * - 批量配置：多序列并行处理的参数设置
   * - 内存布局：灵活的步长配置支持多种数据排列
   */
  struct Arguments : UniversalArgumentsBase
  {
    //
    // 数据成员 - 核心计算参数
    //

    typename EpilogueOutputOp::Params epilogue;  ///< 后处理操作参数（缩放、偏置等）

    // 矩阵数据指针 - 核心计算矩阵
    void const * ptr_A;              ///< 输入矩阵A指针（通常为Query矩阵）
    void const * ptr_B;              ///< 输入矩阵B指针（通常为Key矩阵）
    void const * ptr_C;              ///< 输入矩阵C指针（偏置或累加矩阵）
    void * ptr_D;                    ///< 输出矩阵D指针（注意力分数或结果）
    void const * ptr_sin_cache;      ///< RoPE sin值缓存指针（旋转位置编码）
    void const * ptr_cos_cache;      ///< RoPE cos值缓存指针（旋转位置编码）

    // 批量处理步长 - 多序列并行支持
    int64_t batch_stride_A;          ///< 矩阵A的批量步长（字节）
    int64_t batch_stride_B;          ///< 矩阵B的批量步长（字节）
    int64_t batch_stride_C;          ///< 矩阵C的批量步长（字节）

    // 矩阵布局步长 - 内存访问模式定义
    typename LayoutA::Stride stride_a;        ///< 矩阵A的行/列步长
    typename LayoutB::Stride stride_b;        ///< 矩阵B的行/列步长
    typename LayoutC::Stride stride_c;        ///< 矩阵C的行/列步长
    typename LayoutC::Stride stride_d;        ///< 矩阵D的行/列步长
    typename LayoutC::Stride stride_sin_cache; ///< sin缓存的步长
    typename LayoutC::Stride stride_cos_cache; ///< cos缓存的步长

    // 兼容性步长 - 传统BLAS接口支持
    typename LayoutA::Stride::LongIndex lda;        ///< 矩阵A的leading dimension
    typename LayoutB::Stride::LongIndex ldb;        ///< 矩阵B的leading dimension
    typename LayoutC::Stride::LongIndex ldc;        ///< 矩阵C的leading dimension
    typename LayoutC::Stride::LongIndex ldd;        ///< 矩阵D的leading dimension
    typename LayoutC::Stride::LongIndex ld_sin_cache; ///< sin缓存的leading dimension
    typename LayoutC::Stride::LongIndex ld_cos_cache; ///< cos缓存的leading dimension

    // 稀疏索引数组 - 实现稀疏注意力模式
    int const * ptr_gather_A_indices;        ///< 矩阵A的gather索引（稀疏Query访问）
    int const * ptr_gather_B_indices;        ///< 矩阵B的gather索引（稀疏Key访问）
    int const * ptr_scatter_D_indices;       ///< 矩阵D的scatter索引（稀疏结果写回）
    int const * ptr_gather_sin_cache_indices; ///< sin缓存的gather索引（RoPE稀疏访问）
    int const * ptr_gather_cos_cache_indices; ///< cos缓存的gather索引（RoPE稀疏访问）

    // 稀疏索引批量步长 - 多序列稀疏模式支持
    int64_t batch_stride_gather_A_indices;        ///< A索引数组的批量步长
    int64_t batch_stride_gather_B_indices;        ///< B索引数组的批量步长
    int64_t batch_stride_scatter_D_indices;       ///< D索引数组的批量步长
    int64_t batch_stride_gather_sin_cache_indices; ///< sin索引数组的批量步长
    int64_t batch_stride_gather_cos_cache_indices; ///< cos索引数组的批量步长

    // 序列和注意力头配置 - Transformer模型参数
    int max_seq_len;                         ///< 最大序列长度（内存分配上限）
    int chunk_size;                          ///< 分块大小（计算粒度控制）
    int num_heads;                           ///< 注意力头数量（并行度配置）

    int const *ptr_offset_array;             ///< 偏移数组指针（动态序列长度支持）

    //
    // Methods
    //

    Arguments():
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr),
      ptr_sin_cache(nullptr), ptr_cos_cache(nullptr),
      ptr_gather_A_indices(nullptr),
      ptr_gather_B_indices(nullptr),
      ptr_scatter_D_indices(nullptr),
      ptr_gather_sin_cache_indices(nullptr),
      ptr_gather_cos_cache_indices(nullptr),
      ptr_offset_array(nullptr)
    {}

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void const * ptr_sin_cache,
      void const * ptr_cos_cache,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride stride_a,
      typename LayoutB::Stride stride_b,
      typename LayoutC::Stride stride_c,
      typename LayoutC::Stride stride_d,
      typename LayoutC::Stride stride_sin_cache,
      typename LayoutC::Stride stride_cos_cache,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr,
      int const *ptr_gather_sin_cache_indices = nullptr,
      int const *ptr_gather_cos_cache_indices = nullptr,
      int64_t batch_stride_gather_A_indices = 0,
      int64_t batch_stride_gather_B_indices = 0,
      int64_t batch_stride_scatter_D_indices = 0,
      int64_t batch_stride_gather_sin_cache_indices = 0,
      int64_t batch_stride_gather_cos_cache_indices = 0,
      int max_seq_len = 128*1024,
      int chunk_size = 8,
      int num_heads = 8,
      int const *ptr_offset_array = nullptr)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_sin_cache(ptr_sin_cache), ptr_cos_cache(ptr_cos_cache),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      stride_a(stride_a), stride_b(stride_b), stride_c(stride_c), stride_d(stride_d),
      stride_sin_cache(stride_sin_cache), stride_cos_cache(stride_cos_cache),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices), ptr_gather_sin_cache_indices(ptr_gather_sin_cache_indices),
      ptr_gather_cos_cache_indices(ptr_gather_cos_cache_indices),
      batch_stride_gather_A_indices(batch_stride_gather_A_indices),
      batch_stride_gather_B_indices(batch_stride_gather_B_indices), 
      batch_stride_scatter_D_indices(batch_stride_scatter_D_indices),
      batch_stride_gather_sin_cache_indices(batch_stride_gather_sin_cache_indices),
      batch_stride_gather_cos_cache_indices(batch_stride_gather_cos_cache_indices),
      max_seq_len(max_seq_len),
      chunk_size(chunk_size),
      num_heads(num_heads),
      ptr_offset_array(ptr_offset_array)
    {
      lda = 0;
      ldb = 0;
      ldc = 0;
      ldd = 0;
      ld_sin_cache = 0;
      ld_cos_cache = 0;
      CUTLASS_TRACE_HOST("GemmUniversalBatchGatherIndices::Arguments::Arguments() - problem_size: " << problem_size);
    }

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void const * ptr_sin_cache,
      void const * ptr_cos_cache,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride::LongIndex lda,
      typename LayoutB::Stride::LongIndex ldb,
      typename LayoutC::Stride::LongIndex ldc,
      typename LayoutC::Stride::LongIndex ldd,
      typename LayoutC::Stride::LongIndex ld_sin_cache,
      typename LayoutC::Stride::LongIndex ld_cos_cache,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr,
      int const *ptr_gather_sin_cache_indices = nullptr,
      int const *ptr_gather_cos_cache_indices = nullptr,
      int64_t batch_stride_gather_A_indices = 0,
      int64_t batch_stride_gather_B_indices = 0,
      int64_t batch_stride_scatter_D_indices = 0,
      int64_t batch_stride_gather_sin_cache_indices = 0,
      int64_t batch_stride_gather_cos_cache_indices = 0,
      int max_seq_len = 128*1024,
      int chunk_size = 8,
      int num_heads = 8,
      int const *ptr_offset_array = nullptr
    ):
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_sin_cache(ptr_sin_cache), ptr_cos_cache(ptr_cos_cache),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd), ld_sin_cache(ld_sin_cache), ld_cos_cache(ld_cos_cache),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices), 
      ptr_gather_sin_cache_indices(ptr_gather_sin_cache_indices),
      ptr_gather_cos_cache_indices(ptr_gather_cos_cache_indices),
      batch_stride_gather_A_indices(batch_stride_gather_A_indices),
      batch_stride_gather_B_indices(batch_stride_gather_B_indices), 
      batch_stride_scatter_D_indices(batch_stride_scatter_D_indices),
      batch_stride_gather_sin_cache_indices(batch_stride_gather_sin_cache_indices),
      batch_stride_gather_cos_cache_indices(batch_stride_gather_cos_cache_indices),
      max_seq_len(max_seq_len),
      chunk_size(chunk_size),
      num_heads(num_heads),
      ptr_offset_array(ptr_offset_array)
    {
      stride_a = make_Coord(lda);
      stride_b = make_Coord(ldb);
      stride_c = make_Coord(ldc);
      stride_d = make_Coord(ldd);
      stride_sin_cache = make_Coord(ld_sin_cache);
      stride_cos_cache = make_Coord(ld_cos_cache);
      CUTLASS_TRACE_HOST("GemmUniversalBatchGatherIndices::Arguments::Arguments() - problem_size: " << problem_size);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const
    {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.stride_a, args.stride_b);
      std::swap(args.batch_stride_A, args.batch_stride_B);
      std::swap(args.ptr_gather_A_indices, args.ptr_gather_B_indices);

      return args;
    }
  };


  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : UniversalParamsBase<
    ThreadblockSwizzle,
    ThreadblockShape,
    ElementA,
    ElementB,
    ElementC,
    LayoutA,
    LayoutB>
  {
    using ParamsBase = UniversalParamsBase<
      ThreadblockSwizzle,
      ThreadblockShape,
      ElementA,
      ElementB,
      ElementC,
      LayoutA,
      LayoutB>;

    //
    // Data members
    //

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::SinCosCacheTileIterator::Params params_sin_cache;
    typename Epilogue::SinCosCacheTileIterator::Params params_cos_cache;

    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;
    void * ptr_sin_cache;
    void * ptr_cos_cache;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;

    int * ptr_gather_A_indices;
    int * ptr_gather_B_indices;
    int * ptr_scatter_D_indices;
    int * ptr_gather_sin_cache_indices;
    int * ptr_gather_cos_cache_indices;

    int64_t batch_stride_gather_A_indices;
    int64_t batch_stride_gather_B_indices;
    int64_t batch_stride_scatter_D_indices;
    int64_t batch_stride_gather_sin_cache_indices;
    int64_t batch_stride_gather_cos_cache_indices;

    int max_seq_len;
    int chunk_size;
    int num_heads;

    int * ptr_offset_array;

    //
    // Host dispatch API
    //

    /// Default constructor
    Params() = default;

    /// Constructor
    Params(
      Arguments const &args,  /// GEMM application arguments
      int device_sms,         /// Number of SMs on the device
      int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
    :
      ParamsBase(args, device_sms, sm_occupancy),
      params_A(args.lda ? make_Coord_with_padding<LayoutA::kStrideRank>(args.lda) : args.stride_a, args.chunk_size),
      params_B(args.ldb ? make_Coord_with_padding<LayoutB::kStrideRank>(args.ldb) : args.stride_b),
      params_C(args.ldc ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldc) : args.stride_c),
      params_D(args.ldd ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldd) : args.stride_d),
      params_sin_cache(args.ld_sin_cache ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ld_sin_cache) : args.stride_sin_cache),
      params_cos_cache(args.ld_cos_cache ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ld_cos_cache) : args.stride_cos_cache),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      ptr_sin_cache(const_cast<void *>(args.ptr_sin_cache)),
      ptr_cos_cache(const_cast<void *>(args.ptr_cos_cache)),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      ptr_gather_A_indices(const_cast<int *>(args.ptr_gather_A_indices)),
      ptr_gather_B_indices(const_cast<int *>(args.ptr_gather_B_indices)),
      ptr_scatter_D_indices(const_cast<int *>(args.ptr_scatter_D_indices)),
      ptr_gather_sin_cache_indices(const_cast<int *>(args.ptr_gather_sin_cache_indices)),
      ptr_gather_cos_cache_indices(const_cast<int *>(args.ptr_gather_cos_cache_indices)),
      batch_stride_gather_A_indices(args.batch_stride_gather_A_indices),
      batch_stride_gather_B_indices(args.batch_stride_gather_B_indices),
      batch_stride_scatter_D_indices(args.batch_stride_scatter_D_indices),
      batch_stride_gather_sin_cache_indices(args.batch_stride_gather_sin_cache_indices),
      batch_stride_gather_cos_cache_indices(args.batch_stride_gather_cos_cache_indices),
      max_seq_len(args.max_seq_len),
      chunk_size(args.chunk_size),
      num_heads(args.num_heads),
      ptr_offset_array(const_cast<int *>(args.ptr_offset_array))
    {}

    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      CUTLASS_TRACE_HOST("GemmUniversalBatchGatherIndices::Params::update()");

      // Update input/output pointers
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;
      ptr_sin_cache = const_cast<void *>(args.ptr_sin_cache);
      ptr_cos_cache = const_cast<void *>(args.ptr_cos_cache);

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      this->batch_stride_D = args.batch_stride_D;

      ptr_gather_A_indices = const_cast<int *>(args.ptr_gather_A_indices);
      ptr_gather_B_indices = const_cast<int *>(args.ptr_gather_B_indices);
      ptr_scatter_D_indices = const_cast<int *>(args.ptr_scatter_D_indices);
      ptr_gather_sin_cache_indices = const_cast<int *>(args.ptr_gather_sin_cache_indices);
      ptr_gather_cos_cache_indices = const_cast<int *>(args.ptr_gather_cos_cache_indices);

      batch_stride_gather_A_indices = args.batch_stride_gather_A_indices;
      batch_stride_gather_B_indices = args.batch_stride_gather_B_indices;
      batch_stride_scatter_D_indices = args.batch_stride_scatter_D_indices;
      batch_stride_gather_sin_cache_indices = args.batch_stride_gather_sin_cache_indices;
      batch_stride_gather_cos_cache_indices = args.batch_stride_gather_cos_cache_indices;

      max_seq_len = args.max_seq_len;
      chunk_size = args.chunk_size;
      num_heads = args.num_heads;

      output_op = args.epilogue;

      ptr_offset_array = args.ptr_offset_array;
    }

  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };


public:

  //
  // Host dispatch API
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size)
  {
    CUTLASS_TRACE_HOST("GemmUniversalBatchGatherIndices::can_implement()");

    static int const kAlignmentA = (cute::is_same<LayoutA,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (cute::is_same<LayoutA,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = (cute::is_same<LayoutB,
                                                      layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (cute::is_same<LayoutB,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (cute::is_same<LayoutC,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (cute::is_same<LayoutC,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (cute::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (cute::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (cute::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || cute::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (cute::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (cute::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (cute::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || cute::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (cute::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (cute::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (cute::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || cute::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
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


public:

  //
  // Device-only API
  //

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params const &params,
    SharedStorage &shared_storage)
  {
    GemmUniversalBatchGatherIndices op;
    op(params, shared_storage);
  }


  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    run_with_swizzle(params, shared_storage, threadblock_swizzle);
  }

  /// Executes one GEMM with an externally-provided swizzling function
  CUTLASS_DEVICE
  void run_with_swizzle(Params const &params, SharedStorage &shared_storage, ThreadblockSwizzle& threadblock_swizzle) {

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
    int* ptr_gather_A_indices = params.ptr_gather_A_indices;
    int* ptr_gather_B_indices = params.ptr_gather_B_indices;
    int* ptr_scatter_D_indices = params.ptr_scatter_D_indices;
    ElementC *ptr_sin_cache = static_cast<ElementC *>(params.ptr_sin_cache);
    ElementC *ptr_cos_cache = static_cast<ElementC *>(params.ptr_cos_cache);
    int* ptr_gather_sin_cache_indices = params.ptr_gather_sin_cache_indices;
    int* ptr_gather_cos_cache_indices = params.ptr_gather_cos_cache_indices;
    int* ptr_offset_array = params.ptr_offset_array;

    int offset_value = 0;

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm ||
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() / params.num_heads * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
      if (ptr_gather_A_indices) {
        ptr_gather_A_indices += threadblock_tile_offset.k() * params.batch_stride_gather_A_indices;
      }
      if (ptr_gather_B_indices) {
        ptr_gather_B_indices += threadblock_tile_offset.k() * params.batch_stride_gather_B_indices;
      }

      // get offset value
      if (ptr_offset_array) {
        offset_value = ptr_offset_array[threadblock_tile_offset.k()];
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }

    if (threadblock_tile_offset.m() < offset_value * params.chunk_size / ThreadblockShape::kM) {
      return;
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      ptr_gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      ptr_gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations,
      accumulators,
      iterator_A,
      iterator_B,
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {

      // If performing a reduction via split-K, fetch the initial synchronization
      if (params.grid_tiled_shape.k() > 1) {

        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
      if (ptr_scatter_D_indices) {
        ptr_scatter_D_indices += threadblock_tile_offset.k() * params.batch_stride_scatter_D_indices;
      }
      if (ptr_gather_sin_cache_indices) {
        ptr_gather_sin_cache_indices += threadblock_tile_offset.k() * params.batch_stride_gather_sin_cache_indices;
      }
      if (ptr_gather_cos_cache_indices) {
        ptr_gather_cos_cache_indices += threadblock_tile_offset.k() * params.batch_stride_gather_cos_cache_indices;
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      ptr_scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      ptr_scatter_D_indices
    );

    typename Epilogue::SinCosCacheTileIterator sin_cache_iterator(
      params.params_sin_cache,
      ptr_sin_cache,
      {params.max_seq_len, params.problem_size.n()},
      thread_idx,
      threadblock_offset,
      ptr_gather_sin_cache_indices
    );

    typename Epilogue::SinCosCacheTileIterator cos_cache_iterator(
      params.params_cos_cache,
      ptr_cos_cache,
      {params.max_seq_len, params.problem_size.n()},
      thread_idx,
      threadblock_offset,
      ptr_gather_cos_cache_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op,
      iterator_D,
      accumulators,
      iterator_C,
      sin_cache_iterator,
      cos_cache_iterator);

    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute
    >
struct DefaultMmaSpecial {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::SpecialPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass 

namespace cutlass {
namespace gemm {
namespace kernel {

/**
 * @brief 默认批量聚集索引GEMM配置模板
 * 
 * 这是ShadowKV稀疏注意力机制的核心配置模板，为不同的硬件架构、数据类型
 * 和计算模式提供优化的GEMM内核配置。该模板自动选择最适合的MMA核心、
 * 后处理器和内存访问模式，确保在各种GPU架构上都能获得最佳性能。
 * 
 * 设计理念：
 * - 架构适配：针对Volta/Turing/Ampere等不同架构的专门优化
 * - 类型灵活：支持FP16、BF16、FP32等多种数据类型组合
 * - 稀疏优化：专门针对稀疏注意力模式的内存访问优化
 * - 性能调优：通过模板特化实现最优的计算和内存配置
 */
template <
    /// 矩阵A的元素类型（通常为Query的数据类型）
    typename ElementA,
    /// 矩阵A的布局类型（行主序/列主序）
    typename LayoutA,
    /// 矩阵A的访问对齐粒度（向量化访问优化）
    int kAlignmentA,
    /// 矩阵B的元素类型（通常为Key的数据类型）
    typename ElementB,
    /// 矩阵B的布局类型（行主序/列主序）
    typename LayoutB,
    /// 矩阵B的访问对齐粒度（向量化访问优化）
    int kAlignmentB,
    /// 矩阵C和D的元素类型（输入偏置和输出结果）
    typename ElementC,
    /// 矩阵C和D的布局类型（通常为行主序）
    typename LayoutC,
    /// 内部累加器的元素类型（计算精度控制）
    typename ElementAccumulator,
    /// 操作类别标签（SIMT/TensorOp/WmmaTensorOp）
    typename OperatorClass,
    /// 目标架构标签（Sm70/Sm75/Sm80/Sm86等）
    typename ArchTag,
    /// 线程块级瓦片大小（影响共享内存使用和计算粒度）
    typename ThreadblockShape,
    /// Warp级瓦片大小（影响寄存器使用和并行度）
    typename WarpShape,
    /// 指令级瓦片大小（硬件指令粒度）
    typename InstructionShape,
    /// 后处理输出操作器（缩放、偏置、激活函数等）
    typename EpilogueOutputOp,
    /// 线程块级调度操作器（内存访问模式优化）
    typename ThreadblockSwizzle,
    /// 流水线阶段数（延迟隐藏优化）
    int Stages,
    /// 是否支持SplitK串行归约（大矩阵优化）
    bool SplitKSerial,
    /// GEMM执行的操作类型（乘法、复数乘法等）
    typename Operator,
    /// 共享内存清零选项（异步拷贝优化）
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// 是否对操作数A使用索引数组进行gather操作
    bool GatherA = false,
    /// 是否对操作数B使用索引数组进行gather操作
    bool GatherB = false,
    /// 是否对结果D使用索引数组进行scatter操作
    bool ScatterD = false,
    /// 结果D的排列布局（数据重排优化）
    typename PermuteDLayout = layout::NoPermute,
    /// 操作数A的排列布局（数据重排优化）
    typename PermuteALayout = layout::NoPermute,
    /// 操作数B的排列布局（数据重排优化）
    typename PermuteBLayout = layout::NoPermute,
    /// 模板特化启用条件
    typename Enable = void
>
struct DefaultGemmBatchGatherIndices {

  static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
             "Epilogue in the kernel level must be row major");

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultMmaSpecial<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator, false, SharedMemoryClear, GatherA, GatherB,
      PermuteALayout, PermuteBLayout>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultGatherRopeEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount, ScatterD, PermuteDLayout>::Epilogue;

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value && "Only support RowMajor output C.");

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass


namespace cutlass {
namespace gemm {
namespace kernel {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elemntwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    ///
    typename Enable_ = void
    >
struct DefaultGemmUniversalBatchGatherIndices{

  using DefaultGemmKernel = typename kernel::DefaultGemmBatchGatherIndices<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout,
    PermuteALayout,
    PermuteBLayout
  >::GemmKernel;

  /// Universal kernel without StreamkFeature member type
  template <class SwizzleT, class Enable = void>
  class SelectBase :
    public kernel::GemmUniversalBatchGatherIndices<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      SwizzleT>
  {};

  /// Universal kernel with StreamkFeature member type
  template <class SwizzleT>
  class SelectBase<SwizzleT, typename SwizzleT::StreamkFeature> :
    public kernel::GemmUniversalStreamk<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      SwizzleT>
  {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = SelectBase<ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/*! 
  GemmUniversal is a stateful, reusable GEMM handle.  Once initialized for a given GEMM computation
  (problem geometry and data references), it can be reused across different GEMM problems having the
  geometry.  (Once initialized, details regarding problem geometry and references to workspace memory
  cannot be updated.)

  The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and 
  batched array variants.
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB = ComplexTransform::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout_ = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout_ = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout_ = layout::NoPermute
>
class GemmUniversalBatchGatherIndices : 
  public GemmUniversalBase<
    typename kernel::DefaultGemmUniversalBatchGatherIndices<
      ElementA_,
      LayoutA_,
      TransformA,
      AlignmentA,
      ElementB_,
      LayoutB_,
      TransformB,
      AlignmentB,
      ElementC_,
      LayoutC_,
      ElementAccumulator_,
      OperatorClass_,
      ArchTag_,
      ThreadblockShape_,
      WarpShape_,
      InstructionShape_,
      EpilogueOutputOp_,
      ThreadblockSwizzle_,
      Stages,
      Operator_,
      SharedMemoryClearOption::kNone,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout_,
      PermuteALayout_,
      PermuteBLayout_
    >::GemmKernel
  > {

 public:

  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  using PermuteDLayout = PermuteDLayout_;
  using PermuteALayout = PermuteALayout_;
  using PermuteBLayout = PermuteBLayout_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;

  using Base = GemmUniversalBase<
    typename kernel::DefaultGemmUniversalBatchGatherIndices<
      ElementA_,
      LayoutA_,
      TransformA,
      AlignmentA,
      ElementB_,
      LayoutB_,
      TransformB,
      AlignmentB,
      ElementC_,
      LayoutC_,
      ElementAccumulator_,
      OperatorClass_,
      ArchTag_,
      ThreadblockShape_,
      WarpShape_,
      InstructionShape_,
      EpilogueOutputOp_,
      ThreadblockSwizzle_,
      Stages,
      Operator_,
      SharedMemoryClearOption::kNone,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout_,
      PermuteALayout_,
      PermuteBLayout_
    >::GemmKernel
  >;

  using Arguments = typename Base::Arguments;
  using GemmKernel = typename Base::GemmKernel;
};

} // namespace device
} // namespace gemm
} // namespace cutlass


/////////////////////////////////////////////////////////////////////////////////////////////////
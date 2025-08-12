/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief GatherRopeEpilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction
  operations.

  The shared memory resource is time-sliced across warps.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_hardswish.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_relu0.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/reduction_op.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/permute.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "array_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// GatherRopeEpilogue operator
template <typename Shape_, ///< Shape of threadblock tile (concept: GemmShape)
                           ///< 线程块瓦片的形状（概念：GemmShape）
          typename WarpMmaOperator_, ///< Warp-level MMA operator (concept:
                                     /// gemm::warp::MmaTensorOp)
                                     ///< Warp 级 MMA 操作符（概念：gemm::warp::MmaTensorOp）
          int PartitionsK, ///< Number of partitions of the K dimension
                           ///< K 维度的分区数量
          typename OutputTileIterator_, ///< Tile iterator reading and writing
                                        /// output tensors
                                        ///< 读写输出张量的瓦片迭代器
          typename SinCosCacheTileIterator_, ///< RoPE 的 sin/cos 缓存瓦片迭代器
          typename AccumulatorFragmentIterator_, ///< Fragment iterator
                                                 /// selecting accumulators
                                                 ///< 选择累加器的片段迭代器
          typename WarpTileIterator_,   ///< Warp-scoped tile iterator writing
                                        /// accumulators to SMEM
                                        ///< 将累加器写入共享内存的 Warp 级瓦片迭代器
          typename SharedLoadIterator_, ///< Threadblock-scoped tile iterator
                                        /// loading from SMEM
                                        ///< 从共享内存加载的线程块级瓦片迭代器
          typename OutputOp_,           ///< Output operator
                                        ///< 输出操作符
          typename Padding_, ///< Padding added to SMEM allocation to avoid bank
                             /// conflicts (concept: MatrixShape)
                             ///< 添加到共享内存分配的填充，以避免存储体冲突
          int FragmentsPerPartition =
              1,                 ///< Used to coarsten the epilogue granularity
                                 ///< 用于粗化 epilogue 粒度
          int IterationsUnroll = ///< Used to reduce binary size when epilogue
                                 /// op is large
                                 ///< 当 epilogue 操作较大时用于减少二进制大小
          (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class GatherRopeEpilogue
    : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
                          AccumulatorFragmentIterator_, WarpTileIterator_,
                          Padding_, FragmentsPerPartition>,
      public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_,
                                 AccumulatorFragmentIterator_> {

public:
  using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
                            PartitionsK, AccumulatorFragmentIterator_,
                            WarpTileIterator_, Padding_, FragmentsPerPartition>;

  using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_,
                                          AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using SinCosCacheTileIterator = SinCosCacheTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  /// 每个块的 warp 数量
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  /// 每个块的线程数量
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  /// 每线程累加器瓦片类型
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  /// 数值累加元素类型
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  /// 累加器瓦片片段迭代器使用的片段类型
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  /// Output element
  /// 输出元素类型
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  /// 输出访问大小
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  /// 目标张量的张量引用
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  /// 同步张量的张量引用
  using SyncTensorRef =
      typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  /// 源张量的常量张量引用
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  /// 全局输出迭代器使用的向量类型
  using OutputAccessType = Array<typename OutputTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  /// 共享输出迭代器使用的向量类型
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element,
                                      OutputTileIterator::kElementsPerAccess>;

  /// 共享内存瓦片数量：根据片段迭代次数或 K 分区数确定
  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1
                                        ? Base::kFragmentsPerIteration
                                        : kPartitionsK;

  /// 共享内存指针偏移量：用于在不同瓦片间进行内存分配
  static int constexpr kSmemPointerOffset =
      Base::SharedStorage::StorageShape::kCount / kSmemTiles;

public:
  static_assert(
      SharedLoadIterator::Fragment::kElements ==
          OutputTileIterator::Fragment::kElements,
      "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess,
                "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements %
                  OutputTileIterator::kElementsPerAccess),
                "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1,
                "One of these must be exactly 1.");

public:
  /// 当 epilogue 不需要源数据时的处理方式
  /// 用于只需要累加器数据进行输出操作的场景
  struct SourceAspectNotNeeded {
    /// 构造函数
    CUTLASS_DEVICE
    SourceAspectNotNeeded() {}

    /// 空操作 - 不需要加载源数据
    CUTLASS_DEVICE
    void load() {}

    /// 对输出的每个向量调用输出函数
    /// 仅使用累加器数据进行输出操作，不涉及源数据
    CUTLASS_DEVICE
    void apply_output_operator(
        typename OutputTileIterator::Fragment &output_fragment,
        OutputOp const &output_op,
        typename SharedLoadIterator::Fragment const &aligned_accum_fragment) {
      OutputAccessType *output_frag_ptr =
          reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
          reinterpret_cast<AccumulatorAccessType const *>(
              &aligned_accum_fragment);

      int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
                                      OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i) {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
      }
    }
  };

  /// 当 epilogue 需要源数据时的处理方式
  /// 用于需要累加器数据和源数据进行融合操作的场景（如残差连接）
  struct SourceAspectNeeded {
    /// 源数据瓦片迭代器
    OutputTileIterator source_iterator;

    /// 源数据片段缓存
    typename OutputTileIterator::Fragment source_fragment;

    /// 对输出的每个向量调用输出函数（静态版本）
    /// 同时使用累加器数据和源数据进行输出操作
    CUTLASS_DEVICE
    static void apply_output_operator(
        typename OutputTileIterator::Fragment &output_fragment,
        OutputOp const &output_op,
        typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
        typename OutputTileIterator::Fragment const &source_fragment) {
      OutputAccessType *output_frag_ptr =
          reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
          reinterpret_cast<AccumulatorAccessType const *>(
              &aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr =
          reinterpret_cast<OutputAccessType const *>(&source_fragment);

      int const kOutputOpIterations = OutputTileIterator::Fragment::kElements /
                                      OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i) {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
      }
    }

    /// 构造函数
    /// @param source_iterator 源数据瓦片迭代器
    CUTLASS_DEVICE
    SourceAspectNeeded(OutputTileIterator source_iterator)
        : source_iterator(source_iterator) {
      source_fragment.clear();
    }

    /// 从全局内存加载加数源片段
    /// 用于加载需要与累加器结果进行融合的源数据
    CUTLASS_DEVICE
    void load() {
      source_iterator.load(source_fragment);
      ++source_iterator;
    }

    /// 对输出的每个向量调用输出函数（实例版本）
    /// 调用静态版本的 apply_output_operator
    CUTLASS_DEVICE
    void apply_output_operator(
        typename OutputTileIterator::Fragment &output_fragment,
        OutputOp const &output_op,
        typename SharedLoadIterator::Fragment const &aligned_accum_fragment) {
      apply_output_operator(output_fragment, output_op, aligned_accum_fragment,
                            source_fragment);
    }
  };

private:
  /// 从与输出张量对齐的共享内存加载片段的迭代器
  /// 用于高效地从共享内存读取累加器数据
  SharedLoadIterator shared_load_iterator_;

  /// 线程块内的线程索引
  /// 用于确定当前线程在线程块中的位置
  int thread_idx;

  /// 线程块内的 warp 索引
  /// 用于确定当前 warp 在线程块中的位置
  int warp_idx;

public:
  /// GatherRopeEpilogue 构造函数
  /// 初始化 ShadowKV 稀疏注意力 epilogue 的所有组件
  CUTLASS_DEVICE
  GatherRopeEpilogue(
      typename Base::SharedStorage &shared_storage, ///< 共享存储对象
      int thread_idx, ///< 线程块内的线程 ID
      int warp_idx,   ///< 线程块内的 warp ID
      int lane_idx)   ///< warp 内的线程 ID
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        BaseStreamK(thread_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx),
        thread_idx(thread_idx),
        warp_idx(warp_idx) {}

  /// Aggregates the accumulator sets shared by peer blocks in the global
  /// workspace,
  /// performing epilogue computations, writing to output
  CUTLASS_DEVICE
  void reduce(int peer_idx_begin, int peer_idx_end, int reduce_fragment_idx,
              void *element_workspace,
              OutputOp const &output_op, ///< Output operator
              OutputTileIterator
                  destination_iterator, ///< Tile iterator for destination
              OutputTileIterator source_iterator) ///< Threadblock tile
                                                  /// coordinate in GEMM (in
  /// units of threadblock tiles)
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end,
                        reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    // Initialize/load source-fragment data
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    if (output_op.is_source_needed()) {
      source_iterator += reduce_fragment_idx;
      source_iterator.load(source_fragment);
    }

    // Load fragment from shared memory
    typename SharedLoadIterator::Fragment aligned_accum_fragment;
    shared_load_iterator_.load(aligned_accum_fragment);

    // Add fragments shared by other k partitions
    if (kPartitionsK > 1) {
      plus<typename SharedLoadIterator::Fragment> add_fragments;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kPartitionsK; ++i) {
        typename SharedLoadIterator::Fragment aligned_addend_fragment;
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        shared_load_iterator_.load(aligned_addend_fragment);
        aligned_accum_fragment =
            add_fragments(aligned_accum_fragment, aligned_addend_fragment);
      }
    }

    // Compute the output result
    typename OutputTileIterator::Fragment output_fragment;

    // Apply the output operator
    SourceAspectNeeded::apply_output_operator(
        output_fragment, output_op, aligned_accum_fragment, source_fragment);

    // Store the final result
    destination_iterator += reduce_fragment_idx;
    destination_iterator.store(output_fragment);
  }

  /// Perform the epilogue computations and stream the result to global memory.
  CUTLASS_DEVICE
  void operator()(OutputOp const &output_op, ///< Output operator
                  OutputTileIterator
                      destination_iterator, ///< Tile iterator for destination
                  AccumulatorTile const
                      &accumulators, ///< Complete warp-level accumulator tile
                  SinCosCacheTileIterator sin_cache_iterator,
                  SinCosCacheTileIterator cos_cache_iterator) {
    operator()(output_op, destination_iterator, accumulators,
               SourceAspectNotNeeded(), sin_cache_iterator, cos_cache_iterator);
  }

  /// Perform the epilogue computations and stream the result to global memory.
  /// Implements
  /// two alternative codepaths, depending on whether the output op requires
  /// addend data to be loaded.
  CUTLASS_DEVICE
  void operator()(
      OutputOp const &output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const
          &accumulators, ///< Complete warp-level accumulator tile
      OutputTileIterator source_iterator, ///< Tile iterator for addend source
      SinCosCacheTileIterator sin_cache_iterator,
      SinCosCacheTileIterator cos_cache_iterator) {
    if (output_op.is_source_needed()) {
      operator()(output_op, destination_iterator, accumulators,
                 SourceAspectNeeded(source_iterator), sin_cache_iterator,
                 cos_cache_iterator);
    } else {
      operator()(output_op, destination_iterator, accumulators,
                 SourceAspectNotNeeded(), sin_cache_iterator,
                 cos_cache_iterator);
    }
  }

  /// Perform the epilogue computations and stream the result to global memory.
  /// Implements a
  /// single codepath, regardless of whether the output op requires addend data
  /// to be loaded
  CUTLASS_DEVICE
  void unified(
      OutputOp const &output_op, ///< Output operator
      OutputTileIterator
          destination_iterator, ///< Tile iterator for destination
      AccumulatorTile const
          &accumulators, ///< Complete warp-level accumulator tile
      OutputTileIterator source_iterator) ///< Tile iterator for addend source
  {
    if (!output_op.is_source_needed()) {
      source_iterator.clear_mask();
      __syncthreads(); // Dummy (CUDA 11.0)
    }

    operator()(output_op, destination_iterator, accumulators,
               SourceAspectNeeded(source_iterator));
  }

  template <class Seq> struct acc2smem;

  template <size_t... Seq> struct acc2smem<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void
    helper(AccumulatorFragmentIterator accum_fragment_iterator,
           WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) &&
                     (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };

  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE void
  operator()(OutputOp const &output_op, ///< Output operator
             OutputTileIterator
                 destination_iterator, ///< Tile iterator for destination
             AccumulatorTile const
                 &accumulators, ///< Complete warp-level accumulator tile
             SourceAspect source,
             SinCosCacheTileIterator sin_cache_iterator,
             SinCosCacheTileIterator cos_cache_iterator) {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

//
// Iterate over accumulator tile
//

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      //
      // Load the source
      //

      // TODO: we don't need source load for sure
      source.load();
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::
          push(iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment
          aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      if (kPartitionsK > 1) {
        plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0],
                                                    aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) *
                                                 kSmemPointerOffset);
      }

      //
      // Compute the output result
      //

      // TODO: we don't need this for sure
      typename OutputTileIterator::Fragment output_fragment;
      source.apply_output_operator(output_fragment, output_op,
                                   aligned_accum_fragment[0]);

      // do permulation and negation
      // shuffle_odd_even_negate<typename OutputTileIterator::Fragment>
      //     shuffle_functor;
      // typename OutputTileIterator::Fragment permutated_fragment =
      //     shuffle_functor(output_fragment);

      // load sin cos cache
      // typename SinCosCacheTileIterator::Fragment sin_fragment;
      // typename SinCosCacheTileIterator::Fragment cos_fragment;
      // sin_cache_iterator.load(sin_fragment);
      // cos_cache_iterator.load(cos_fragment);
      // ++sin_cache_iterator;
      // ++cos_cache_iterator;

      // do rope
      // multiplies<typename OutputTileIterator::Fragment> mul_functor;
      // multiply_add<typename OutputTileIterator::Fragment> mul_add_functor;
      // auto right = mul_functor(permutated_fragment, sin_fragment);
      // auto result = mul_add_functor(output_fragment, cos_fragment, right);

      //
      // Store the final result
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute,
          conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
          int Rank = 4>
struct DefaultGatherRopeEpilogueTensorOp {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;
  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  //
  // Thread map
  //

  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          Shape, typename WarpMmaTensorOp::Shape, kPartitionsK, ElementOutput,
          kElementsPerAccess>::Type;

  static bool const UseCUDAStore =
      platform::is_same<ElementOutput, double>::value;

  using PackedOutputTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, ScatterD, PermuteDLayout,
          UseCUDAStore>;

  using StridedOutputTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIteratorConv<
          OutputTileThreadMap, ElementOutput, ScatterD, PermuteDLayout,
          UseCUDAStore, kRank>;

  using OutputTileIterator = typename platform::conditional<
      StrideSupport == cutlass::conv::StrideSupport::kUnity,
      PackedOutputTileIterator, StridedOutputTileIterator>::type;

  using SinCosCacheTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, true, PermuteDLayout,
          UseCUDAStore>;

  using AccumulatorFragmentIterator = typename platform::conditional<
      is_complex<ElementOutput>::value,
      cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC, LayoutC>,
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC,
          LayoutC>>::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
      ElementOutput, ElementAccumulator, kElementsPerAccess, Shape,
      typename WarpMmaTensorOp::Shape,
      typename WarpMmaTensorOp::Policy::Operator::Shape,
      typename OutputTileThreadMap::CompactedThreadMap>;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added
  using Padding =
      cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

  static int const kFragmentsPerIteration =
      (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::GatherRopeEpilogue<
      Shape, WarpMmaTensorOp, kPartitionsK, OutputTileIterator,
      SinCosCacheTileIterator, AccumulatorFragmentIterator, WarpTileIterator,
      SharedLoadIterator, OutputOp, Padding, kFragmentsPerIteration>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
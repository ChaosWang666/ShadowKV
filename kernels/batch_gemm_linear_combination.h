/***************************************************************************************************
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
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
  \brief 执行 epilogue 中线性组合操作的函数对象
  
  中文详细说明：
  这个文件实现了 ShadowKV 项目中批量 GEMM 操作的线性组合功能。
  线性组合是 GEMM epilogue 阶段的核心操作，负责将累加器结果转换为最终输出。
  
  主要功能：
  1. 标准线性组合：D = alpha * accumulator + beta * C
  2. 批量操作支持：为每个批次提供独立的 alpha/beta 参数
  3. 按通道缩放：支持每个通道独立的缩放因子
  4. 数值转换：支持不同精度间的安全转换
  5. 内存优化：减少中间结果的内存占用
  
  性能特点：
  - 向量化操作：一次处理多个元素
  - 融合计算：减少内存访问次数
  - 数值稳定性：支持多种舍入模式
  - 灵活的参数传递：支持常量、指针和数组形式的参数
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source
///
/**
 * @brief 批量 GEMM 线性组合操作类
 * 
 * 这个类实现了 ShadowKV 稀疏注意力机制中的核心线性组合操作。
 * 它负责将 GEMM 累加器的结果转换为最终的输出格式。
 * 
 * 数学公式：D = alpha * accumulator + beta * C
 * 其中：
 * - D: 输出张量
 * - accumulator: GEMM 计算的累加器结果
 * - C: 源张量（可选，用于残差连接等）
 * - alpha, beta: 缩放因子
 */
template <
  typename ElementOutput_,                             ///< 用于加载和存储张量的数据类型
  int Count,                                           ///< 每次操作计算的元素数量
                                                       ///< 通常是 128/sizeof_bits<ElementOutput_>
                                                       ///< 但当数据不足时也使用 64 或 32
  typename ElementAccumulator_ = ElementOutput_,       ///< 累加器数据类型
  typename ElementCompute_ = ElementOutput_,           ///< 用于计算线性组合的数据类型
  ScaleType::Kind Scale = ScaleType::Default,          ///< 控制 Alpha 和 Beta 缩放的类型
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, ///< 浮点舍入方式
  typename ElementSource_ = ElementOutput_             ///< 源张量元素类型
>
class BatchGEMMLinearCombination {
public:

  /// 输出元素类型
  using ElementOutput = ElementOutput_;
  /// 源张量元素类型
  using ElementSource = ElementSource_;
  /// 累加器元素类型
  using ElementAccumulator = ElementAccumulator_;
  /// 计算元素类型
  using ElementCompute = ElementCompute_;
  /// 标量元素类型（与计算类型相同）
  using ElementScalar = ElementCompute;
  /// C 矩阵元素类型（与源类型相同）
  using ElementC = ElementSource_;
  /// D 矩阵元素类型（与输出类型相同）
  using ElementD = ElementOutput_;

  /// 每次操作处理的元素数量
  static int const kCount = Count;
  /// 缩放类型
  static const ScaleType::Kind kScale = Scale;
  /// 输出片段类型
  using FragmentOutput = Array<ElementOutput, kCount>;
  /// 源片段类型
  using FragmentSource = Array<ElementSource, kCount>;
  /// 累加器片段类型
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  /// 计算片段类型
  using FragmentCompute = Array<ElementCompute, kCount>;

  /// 浮点舍入方式
  static FloatRoundStyle const kRound = Round;

  /// 线性组合的参数结构体
  /// 支持多种参数传递方式：常量、指针、批量指针数组
  struct Params
  {
    ElementCompute alpha;                         ///< 累加器缩放因子
    ElementCompute beta;                          ///< 源张量缩放因子
    ElementCompute const *alpha_ptr;              ///< 累加器标量指针 - 如果非空，从内存加载
    ElementCompute const *beta_ptr;               ///< 源标量指针 - 如果非空，从内存加载
    ElementCompute const* const* alpha_ptr_array; ///< 每个组/批次的累加器标量指针数组
    ElementCompute const* const* beta_ptr_array;  ///< 每个组/批次的源标量指针数组

    /// 默认构造函数：alpha=1, beta=0（仅累加器输出）
    CUTLASS_HOST_DEVICE
    Params():
      alpha(ElementCompute(1)),
      beta(ElementCompute(0)),
      alpha_ptr(nullptr),
      beta_ptr(nullptr),
      alpha_ptr_array(nullptr),
      beta_ptr_array(nullptr) { }

    /// 常量参数构造函数：使用固定的 alpha 和 beta 值
    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta
    ):
      alpha(alpha), beta(beta),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }

    /// 仅 alpha 参数构造函数：beta 设为 0
    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ):
      alpha(alpha), beta(0),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr
    ):
      alpha(0), beta(0),
      alpha_ptr(alpha_ptr), beta_ptr(beta_ptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ):
      alpha(0), beta(0),
      alpha_ptr(alpha_ptr), beta_ptr(nullptr),
      alpha_ptr_array(nullptr), beta_ptr_array(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const* const* alpha_ptr_array,
      ElementCompute const* const* beta_ptr_array
    ):
      alpha(0), beta(0),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(alpha_ptr_array), beta_ptr_array(beta_ptr_array) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const* const* alpha_ptr_array
    ):
      alpha(0), beta(0),
      alpha_ptr(nullptr), beta_ptr(nullptr),
      alpha_ptr_array(alpha_ptr_array), beta_ptr_array(nullptr) { }
  };

private:

  /// 数据成员
  /// 存储当前实例使用的缩放因子

  /// 累加器缩放因子
  ElementCompute alpha_;
  /// 源张量缩放因子
  ElementCompute beta_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  BatchGEMMLinearCombination(Params const &params, int group_idx = 0) {
    if (params.alpha_ptr_array != nullptr && params.alpha_ptr_array[group_idx] != nullptr) {
      alpha_ = *(params.alpha_ptr_array[group_idx]);
    }
    else if (params.alpha_ptr != nullptr) {
      alpha_ = *params.alpha_ptr;
    }
    else {
      alpha_ = params.alpha;
    }
    if (params.beta_ptr_array != nullptr && params.beta_ptr_array[group_idx] != nullptr) {
      beta_ = *(params.beta_ptr_array[group_idx]);
    }
    else if (params.beta_ptr != nullptr) {
      beta_ = *params.beta_ptr;
    }
    else {
      beta_ = params.beta;
    }
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling) return true;

    if (Scale == ScaleType::OnlyAlphaScaling) return false;

    if (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator,
      FragmentSource const &source) const {

    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling)
      intermediate = converted_source;
    else
      intermediate = mul_add_source(beta_, converted_source);                             // X =  beta * C + uniform

    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  #if 0
  {
    NumericConverter<ElementOutput, ElementCompute, Round> scale_converter;
    ElementOutput alpha_out = scale_converter(alpha_);

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> downcast_converter;
    // NumericArrayConverter<ElementOutput, ElementCompute, 1, Round> scalecast_converter;

    // NumericArrayConverter<ElementSoftmaxCompute, ElementOutput, kElementsPerAccess> source_converter;


    // FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    FragmentOutput converted_accumulator_out = downcast_converter(accumulator);

    // if (Scale == ScaleType::Nothing)
    //   return destination_converter(converted_accumulator);

    // Perform binary operations
    FragmentCompute intermediate;
    multiplies<FragmentCompute> mul_accumulator;

    FragmentOutput intermediate_out;
    multiplies<FragmentOutput> mul_accumulator_out;

    // intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    // FragmentOutput alpha_out = scalecast_converter(alpha_);
    intermediate_out = mul_accumulator_out(alpha_out, converted_accumulator_out);

    // return destination_converter(intermediate);
    return intermediate_out;
  }
#endif
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const &accumulator) const {
    // printf(".........\n");
    // alpha converter
    NumericConverter<ElementOutput, ElementCompute, Round> alpha_converter;
    ElementOutput alpha_out = alpha_converter(alpha_);
    // Convert source to interal compute numeric type
    // NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    // FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    FragmentOutput converted_accumulator_out = destination_converter(accumulator);

    // if (Scale == ScaleType::Nothing)
    //   return destination_converter(converted_accumulator);

    // Perform binary operations
#if 0
    FragmentCompute intermediate;
    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(alpha_, converted_accumulator);    // D = alpha * Accum

    return destination_converter(intermediate);
#else
    FragmentOutput intermediate;
    multiplies<FragmentOutput> mul_accumulator;
#endif
    intermediate = mul_accumulator(alpha_out, converted_accumulator_out);    // D = alpha * Accum

    return intermediate;
  }

  //
  // Specializations for scalar (for use with cute::collective::DefaultEpilogue)
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    [[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type

    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;
    multiply_add<ElementCompute> madd;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = source_converter(source);
    }
    else {
      intermediate = multiply(beta_, source);                            // X =  beta * C + uniform
    }

    intermediate = madd(alpha_, converted_accumulator, intermediate);    // D = alpha * Accum + X
    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;
    ElementCompute converted_accumulator = accumulator_converter(accumulator);

    // Convert to destination numeric type
    if constexpr (Scale == ScaleType::Nothing) {
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiplies<ElementCompute> multiply;

    intermediate = multiply(alpha_, accumulator);    // D = alpha * Accum
    return destination_converter(intermediate);
  }
};

/// Applies a linear combination operator to an array of elements.
///
/// D = vector_alpha * accumulator + (optional) vector_beta/scalar_beta * source
///
template <
  typename ElementOutput_,            ///< Data type used to load and store tensors
  int Count,                          ///< Number of elements computed per operation.
  typename ElementAccumulator_,       ///< Accumulator data type
  typename ElementCompute_,           ///< Data type used to compute linear combination
  FloatRoundStyle Round,
  typename ElementSource_
>
class BatchGEMMLinearCombination<ElementOutput_,
                        Count,
                        ElementAccumulator_,
                        ElementCompute_,
                        ScaleType::PerChannelScaling,
                        Round,
                        ElementSource_> {
public:

  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = ScaleType::PerChannelScaling;
  static constexpr bool IsPerChannelScalingSupported = true;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params
  {
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator vector
    ElementCompute const *beta_ptr;        ///< pointer to source vector
    ElementCompute beta;                   ///< scales source tensor

    CUTLASS_HOST_DEVICE
    Params():
      alpha_ptr(nullptr),
      beta_ptr(nullptr),
      beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr
    ):
      alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ):
      alpha_ptr(alpha_ptr), beta_ptr(nullptr), beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute beta
    ):
      alpha_ptr(alpha_ptr), beta_ptr(nullptr), beta(beta) { }

  };

private:

  //
  // Data members
  //

  ElementCompute const* beta_ptr_ = nullptr;
  ElementCompute beta_ = 0;

public:

  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  BatchGEMMLinearCombination(Params const& params) {
    if (params.beta_ptr) {
      beta_ptr_ = params.beta_ptr;
    }
    else {
      beta_ = params.beta;
    }
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ptr_ != nullptr || beta_ != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  bool is_beta_vector() const {
    return beta_ptr_ != nullptr;
  }

  /// Computes linear scaling with source: D = vector_alpha * accumulator + vector_beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentSource const& source,
      FragmentCompute const& valpha,
      FragmentCompute const& vbeta) const {
    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate = mul_add_source(vbeta, converted_source);                             // X = vector_beta * C + uniform

    intermediate = mul_add_accumulator(valpha, converted_accumulator, intermediate);    // D = vector_alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling with source: D = vector_alpha * accumulator + scalar_beta(from host) * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentSource const& source,
      FragmentCompute const& valpha) const {
    // Convert source to internal compute numeric type
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;

    multiplies<FragmentCompute> mul_add_source;
    multiply_add<FragmentCompute> mul_add_accumulator;


    intermediate = mul_add_source(beta_, converted_source);                           // X =  scalar_beta * C + uniform

    intermediate = mul_add_accumulator(valpha, converted_accumulator, intermediate);    // D = vector_alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = vector_alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentCompute const& valpha) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;
    multiplies<FragmentCompute> mul_accumulator;

    intermediate = mul_accumulator(valpha, converted_accumulator);    // D = vector_alpha * Accum

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
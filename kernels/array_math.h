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
 * @file array_math.h
 * @brief 数组数学运算工具函数
 * 
 * 本文件定义了用于 RoPE (Rotary Position Embedding) 计算的数组操作函数。
 * 主要功能是实现复数旋转所需的元素重排和符号变换。
 * 
 * RoPE 的核心思想：
 * - 将嵌入向量的相邻元素对视为复数的实部和虚部
 * - 通过复数乘法实现位置相关的旋转变换
 * - 需要特定的元素重排和符号操作来实现高效计算
 * 
 * 数学原理：
 * 对于复数 z = a + bi 和旋转因子 w = cos(θ) + i*sin(θ)
 * z * w = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))
 * 
 * 实现策略：
 * - shuffle_odd_even_negate: 实现 [a, b] -> [b, -a] 的变换
 * - 配合 cos/sin 值进行向量化的复数乘法运算
 * 
 * 中文详细说明：
 * 这个文件是 ShadowKV 项目中用于实现旋转位置编码（RoPE）的核心数学工具。
 * RoPE 是一种先进的位置编码方法，通过将向量元素视为复数并应用旋转变换来编码位置信息。
 * 这种方法相比传统的绝对位置编码具有更好的外推能力和相对位置感知能力。
 */

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
namespace cutlass {

/**
 * @brief 奇偶元素重排并取负数的模板声明
 * 
 * 这是 RoPE 计算中的核心操作，用于实现复数乘法的虚部计算。
 * 将输入数组的相邻元素对进行特定的重排和符号变换。
 * 
 * 中文说明：
 * 这个模板函数是实现 RoPE 旋转位置编码的关键组件。在 RoPE 中，我们需要将
 * 向量的相邻元素对视为复数，然后进行旋转变换。这个函数实现了复数乘法中
 * 虚部计算所需的元素重排和符号变换操作。
 */
template <typename T>
struct shuffle_odd_even_negate;

/**
 * @brief 针对 CUTLASS Array 类型的奇偶重排特化实现
 * 
 * 实现 RoPE 复数旋转计算中的关键变换操作：
 * - 输入：[x0, x1, x2, x3, ...] (相邻元素对表示复数)
 * - 输出：[x1, -x0, x3, -x2, ...] (为虚部计算准备)
 * 
 * 数学含义：
 * 对于复数对 (a, b)，变换为 (b, -a)
 * 这样可以通过向量化操作实现：
 * - 实部：a*cos - b*sin = [a, b] · [cos, sin] - [b, -a] · [sin, cos]
 * - 虚部：a*sin + b*cos = [a, b] · [sin, cos] + [b, -a] · [cos, sin]
 * 
 * @tparam T 数组元素类型 (通常是 bfloat16 或 float)
 * @tparam N 数组大小 (必须是偶数)
 * 
 * 中文详细说明：
 * 这个特化实现是 RoPE 算法的核心数学操作。在 RoPE 中，我们将嵌入向量的
 * 相邻元素对 (x0, x1), (x2, x3), ... 视为复数 x0+i*x1, x2+i*x3, ...
 * 为了高效地计算复数乘法 (a+bi) * (cos+i*sin)，我们需要进行元素重排。
 * 这个函数将 [a, b] 重排为 [b, -a]，使得我们可以用向量化的乘法和加法
 * 来计算复数乘法的实部和虚部，大大提高了计算效率。
 */
template <typename T, int N>
struct shuffle_odd_even_negate<Array<T, N>> {
  // 静态断言：确保数组大小为偶数，因为我们处理的是复数对
  static_assert(N % 2 == 0 && "Only can shuffle 2K elements.");

  /**
     * @brief 执行奇偶重排和取负操作
     * 
     * @param rhs 输入数组
     * @return 重排后的数组
     * 
     * 变换规则：
     * - result[2*i] = -rhs[2*i+1]  (偶数位置：取奇数位置元素的负值)
     * - result[2*i+1] = rhs[2*i]   (奇数位置：取偶数位置元素)
     * 
     * 中文说明：
     * 这个函数实现了 RoPE 复数旋转所需的关键数据重排操作。对于输入的
     * 向量 [a0, b0, a1, b1, ...]，它会输出 [-b0, a0, -b1, a1, ...]。
     * 这种重排使得我们可以用简单的向量运算来实现复数乘法，避免了
     * 复杂的索引计算，提高了 GPU 上的执行效率。
     */
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const & rhs) const {
    Array<T, N> result;

    // 展开循环以提高性能 - 编译器会将循环完全展开，消除分支开销
    // 这对于 GPU 计算特别重要，因为可以最大化并行度和内存带宽利用率
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      // 实现 RoPE 复数旋转的关键变换：[a, b] -> [b, -a]
      // 这种重排为后续的向量化复数乘法做准备
      result[2 * i + 1] = rhs[2 * i];      // 奇数位置 = 原偶数位置 (复数实部)
      result[2 * i] = -rhs[2 * i + 1];     // 偶数位置 = 原奇数位置的负值 (复数虚部取负)
    }

    return result;
  }
};

} // cutlass
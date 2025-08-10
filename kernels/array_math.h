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
   */
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const & rhs) const {
    Array<T, N> result;

    // 展开循环以提高性能
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result[2 * i + 1] = rhs[2 * i];      // 奇数位置 = 原偶数位置
      result[2 * i] = -rhs[2 * i + 1];     // 偶数位置 = 原奇数位置的负值
    }

    return result;
  }
};

} // cutlass
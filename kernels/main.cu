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
 * @file main.cu
 * @brief ShadowKV CUDA 内核模块的 Python 绑定入口文件
 * 
 * 本文件使用 PyBind11 将 ShadowKV 的核心 CUDA 内核函数暴露给 Python，
 * 主要包含以下几类优化内核：
 * 1. 稀疏注意力相关的 gather-copy 操作
 * 2. 旋转位置编码 (RoPE) 的高效实现
 * 3. 批量 GEMM 和 Softmax 融合操作
 * 
 * 这些内核是 ShadowKV 高效推理的核心组件，通过 CUDA 加速实现了
 * 内存优化的稀疏注意力计算和 KV 缓存管理。
 */

#include <torch/extension.h>
#include "functions.h"

/**
 * @brief PyBind11 模块定义，将 CUDA 内核函数暴露给 Python
 * 
 * 该模块包含 ShadowKV 的所有核心 CUDA 内核函数，按功能分类如下：
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // === 稀疏注意力 Gather-Copy 操作 ===
    // 基础的 gather-copy 操作，用于从 CPU 内存收集数据到 GPU
    m.def("gather_copy", &gather_copy, "基础 Gather-Copy 操作 (CUDA)");
    
    // GPU 到 GPU 的 gather-copy，带偏移量优化，用于 key 张量
    m.def("gather_copy_d2d_with_offsets", &gather_copy_d2d_with_offsets, "GPU间带偏移量的 Gather-Copy，用于 keys (CUDA)");
    
    // 重排序 keys 并计算偏移量，为后续 gather-copy 做准备
    m.def("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets, "重排序 keys 并计算偏移量 (CUDA)");
    
    // 带偏移量的 gather-copy，支持更高效的内存访问模式
    m.def("gather_copy_with_offsets", &gather_copy_with_offsets, "带偏移量的 Gather-Copy (CUDA)");
    
    // === 旋转位置编码 (RoPE) 操作 ===
    // 基础的 RoPE 应用，分别使用 cos 和 sin 张量
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "应用旋转位置编码 (CUDA)");
    
    // 优化版本，使用融合的 cos_sin 张量
    m.def("apply_rotary_pos_emb_new", &apply_rotary_pos_emb_new, "应用旋转位置编码-新版本 (CUDA)");
    
    // 进一步优化版本，支持分块处理
    m.def("apply_rotary_pos_emb_new_v2", &apply_rotary_pos_emb_new_v2, "应用旋转位置编码-v2版本 (CUDA)");
    
    // 带缓存推送的 RoPE，用于增量推理
    m.def("apply_rotary_pos_emb_push_cache", &apply_rotary_pos_emb_push_cache, "应用 RoPE 并推送缓存 (CUDA)");
    
    // 优化的缓存推送版本
    m.def("apply_rotary_pos_emb_push_cache_opt", &apply_rotary_pos_emb_push_cache_opt, "应用 RoPE 并推送缓存-优化版 (CUDA)");
    
    // 专门为 GLM 模型优化的版本
    m.def("apply_rotary_pos_emb_push_cache_opt_glm", &apply_rotary_pos_emb_push_cache_opt_glm, "应用 RoPE 并推送缓存-GLM优化版 (CUDA)");
    
    // === 批量矩阵运算操作 ===
    // 批量 gather GEMM，结合稀疏注意力的矩阵乘法
    m.def("batch_gather_gemm", &batch_gather_gemm, "批量 Gather GEMM (CUDA)");
    
    // 批量 GEMM 与 Softmax 融合操作
    m.def("batch_gemm_softmax", &batch_gemm_softmax, "批量 GEMM Softmax (CUDA)");
}
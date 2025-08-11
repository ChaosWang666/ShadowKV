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
 * @file gather_copy.cu
 * @brief ShadowKV 稀疏注意力的 Gather-Copy 内核实现
 * 
 * 本文件实现了 ShadowKV 稀疏注意力机制中的核心 gather-copy 操作，
 * 主要功能包括：
 * 1. 基础的 gather-copy：从 CPU 内存收集指定位置的数据到 GPU
 * 2. GPU 间的 gather-copy：在 GPU 内存中进行高效的数据重排
 * 3. 带偏移量的优化版本：使用预计算偏移量提高内存访问效率
 * 4. 位置 ID 重排序：为后续操作优化数据布局
 * 
 * 这些操作是 ShadowKV 实现稀疏注意力的关键，通过只收集和计算
 * 重要的 KV 对，大幅减少内存使用和计算量。
 */

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

// 排序偏移量标志，用于控制排序行为
#define SORT_OFFSET 1

// 默认块大小定义，影响内核的并行度和内存访问模式
#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#include "copy.cuh"
#include "functions.h"
#include "map.cuh"

// 根据块大小选择合适的数据类型，优化内存访问
// 128 线程块使用 int4（16字节），256 线程块使用 int2（8字节）
#if BLOCK_SIZE_CP == 128
#define PTYPE int4  // 16字节向量化访问，适合较小的线程块
#endif

#if BLOCK_SIZE_CP == 256
#define PTYPE int2  // 8字节向量化访问，适合较大的线程块
#endif

/**
 * @brief 基础的 Gather-Copy 操作实现
 * 
 * 从 CPU 内存中根据位置 ID 收集 value 数据到 GPU 缓存中。
 * 这是 ShadowKV 稀疏注意力的核心操作，通过只收集重要的数据
 * 来减少内存使用和提高计算效率。
 * 
 * @param values CPU 上的 value 张量（BFloat16 格式）
 * @param v_cache_buffer GPU 上的 value 缓存缓冲区（BFloat16 格式）
 * @param position_ids 位置 ID 张量，指定要收集的数据位置
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param cpu_v_length CPU 上 value 的长度
 * @param gpu_v_length GPU 上 value 缓存的长度
 * @param map_size 映射表大小，默认为 256
 */
void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size = 256)
{
    // 设置 CUDA 内核参数
    int blockSize = BLOCK_SIZE_CP;  // 每个线程块的线程数
    int numBlocks = batch_size * heads;  // 线程块总数，每个注意力头一个块
    
    // 计算共享内存需求：复制缓冲区 + 映射表，必须小于 160KB
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4;

    // 将 BFloat16 数据指针转换为向量化类型指针，提高内存访问效率
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 256><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }  else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 128><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 512><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gahter_copy_fixed_start_end<PTYPE, int64_t, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gahter_copy_fixed_start_end<PTYPE, int64_t, 1024><<<numBlocks, blockSize, maxSMBytes>>>(
        values_ptr,
        v_cache_buffer_ptr,
        cpu_v_length, 
        gpu_v_length, 
        position_ids.data_ptr<int64_t>(),
        0/*assume no hit*/, 
        map_size);
    }
}

/**
 * @brief GPU 到 GPU 的 Key 数据 Gather-Copy 操作
 * 
 * 在 GPU 内存中进行高效的 key 数据收集和重排，使用预计算的偏移量
 * 来优化内存访问模式。该函数专门用于 key 张量的设备间拷贝，
 * 支持 ShadowKV 稀疏注意力中的动态 key 选择。
 * 
 * @param keys GPU 上的 key 张量（BFloat16 格式）
 * @param offsets 预计算的偏移量数组，由 reorder_keys_and_compute_offsets 计算
 * @param cnts 计数数组，用于分离 D2D 和 H2D 操作
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param gpu_v_length GPU 上数据的长度
 * @param gpu_v_offset GPU 数据的起始偏移量
 * @param gpu_v_stride GPU 数据的内存步长
 * @param map_size 映射表大小，默认为 256
 */
void gather_copy_d2d_with_offsets(
    torch::Tensor keys, 
    torch::Tensor offsets,          // 输入：预计算的偏移量数组 (numBlocks*256)
    torch::Tensor cnts,             // 输入：计数数组 (numBlocks)
    int batch_size, int heads, 
    int gpu_v_length, 
    int gpu_v_offset, 
    int gpu_v_stride, 
    int map_size = 256)
{
    // 设置 CUDA 内核参数
    int blockSize = BLOCK_SIZE_CP;  // 每个线程块的线程数
    int numBlocks = batch_size * heads;  // 线程块总数
    // 计算共享内存需求：复制缓冲区 + 映射表 + 额外空间，必须小于 160KB
    int maxSMBytes = CPY_SIZE*2*1024 + map_size*4 + sizeof(PTYPE);

    // 将 BFloat16 数据指针转换为向量化类型指针，提高内存访问效率
    PTYPE* keys_ptr = reinterpret_cast<PTYPE*>(keys.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());

    // this only needs to run once
    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);    
    } else if (map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0 /*start*/, 
        cnts_ptr);
    } else if (map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_d2d<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_d2d<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes>>>(
        keys_ptr, 
        nullptr,
        gpu_v_length, 
        gpu_v_offset,
        gpu_v_stride,
        offsets_ptr,
        0, /*start*/
        cnts_ptr);
    }
}

/**
 * @brief 重排序位置 ID 并计算偏移量和计数
 * 
 * 通过计算缓存命中/未命中来重排序位置 ID，并计算用于后续
 * gather_copy_with_offsets 操作的偏移量和计数。这是 ShadowKV
 * 稀疏注意力中的关键预处理步骤，确保高效的数据收集。
 * 
 * 必须在调用 gather_copy_with_offsets 之前调用此函数。
 * 
 * @param cached_pos_ids 缓存的位置 ID（输入输出，int64_t 类型）
 * @param cur_pos_ids 当前传入的位置 ID（输入，int64_t 类型）
 * @param offsets 输出的偏移量数组，用于 gather_copy_with_offsets
 * @param cnts 输出的计数数组，用于分离 D2D 和 H2D 操作
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param map_size 映射表大小，默认为 256
 */
void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // 输入输出：缓存的位置 ID，同时作为重排序后的位置 ID 输出
    torch::Tensor cur_pos_ids,    // 输入：当前传入的位置 ID (int64_t)
    torch::Tensor offsets,        // 输出：用于 gather_copy_with_offsets 的偏移量数组 (int)
    torch::Tensor cnts,           // 输出：用于分离 D2D 和 H2D 操作的计数数组 (int)
    int batch_size, int heads, 
    int map_size = 256)
{
    // 设置 CUDA 内核参数：每个线程块处理一个映射表
    int blockSize = map_size;  // 线程块大小等于映射表大小
    int numBlocks = batch_size * heads;  // 线程块总数

    // 获取数据指针
    int64_t* cached_pos = cached_pos_ids.data_ptr<int64_t>();
    int64_t* cur_pos = cur_pos_ids.data_ptr<int64_t>();
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());


    if(map_size == 256) {
        reorder_keys_and_mixed_offsets<int64_t, 256, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 128) {
        reorder_keys_and_mixed_offsets<int64_t, 128, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 512) {
        reorder_keys_and_mixed_offsets<int64_t, 512, 2048><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 1024) {
        reorder_keys_and_mixed_offsets<int64_t, 1024, 4096><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    }
}

/**
 * @brief 带偏移量的 Gather-Copy 操作
 * 
 * 使用预计算的偏移量进行高效的 value 数据收集，支持 CPU 到 GPU
 * 的异步数据传输。这是 ShadowKV 中最重要的内存优化操作之一，
 * 实现了稀疏注意力中的核心数据收集逻辑。
 * 
 * 必须在调用 reorder_keys_and_compute_offsets 之后调用此函数。
 * 
 * @param values CPU 上的 value 张量（BFloat16 格式）
 * @param v_cache_buffer GPU 上的 value 缓存缓冲区（输入输出）
 * @param temp 临时 GPU 内存，大小与单层 v_cache_buffer 相同
 * @param offsets 预计算的偏移量数组，由 reorder_keys_and_compute_offsets 计算
 * @param cnts 计数数组，由 reorder_keys_and_compute_offsets 计算
 * @param signals 内部信号数组，全零初始化
 * @param batch_size 批次大小
 * @param heads 注意力头数
 * @param cpu_v_length CPU 上 value 的长度
 * @param gpu_v_length GPU 上 value 缓存的长度
 * @param gpu_v_offset GPU value 缓存的起始偏移量
 * @param gpu_v_stride GPU value 缓存的内存步长
 * @param map_size 映射表大小，默认为 256
 */
void gather_copy_with_offsets(
    torch::Tensor values,           // 输入：CPU 上的 value 张量
    torch::Tensor v_cache_buffer,   // 输入输出：GPU 上的 value 缓存缓冲区
    torch::Tensor temp,             // 临时 GPU 内存，大小与单层 v_cache_buffer 相同
    torch::Tensor offsets,          // 输入：预计算的偏移量数组 (numBlocks*256)
    torch::Tensor cnts,             // 输入：计数数组 (numBlocks)
    torch::Tensor signals,          // 内部信号数组，全零初始化 (numBlocks)
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size = 256)
{
    // 设置 CUDA 内核参数
    int blockSize = BLOCK_SIZE_CP;  // 每个线程块的线程数
    int numBlocks = batch_size * heads * 2; // 线程块总数，使用双缓冲策略
    // 计算共享内存需求：复制缓冲区 + 映射表 + 额外空间，必须小于 160KB
    int maxSMBytes = CPY_SIZE * 2 * 1024 + map_size * 4 + sizeof(PTYPE);

    // 将 BFloat16 数据指针转换为向量化类型指针，提高内存访问效率
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());
    PTYPE* temp_ptr = reinterpret_cast<PTYPE*>(temp.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());
    unsigned int* signals_ptr = reinterpret_cast<unsigned int*>(signals.data_ptr<int32_t>());
    
    // 获取当前 CUDA 流，支持异步操作
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }else if(map_size == 512) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 512><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    } else if(map_size == 1024) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 1024><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            values_ptr, 
            temp_ptr, 
            v_cache_buffer_ptr, 
            cpu_v_length, 
            gpu_v_length,
            gpu_v_offset,
            gpu_v_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr);
    }
}

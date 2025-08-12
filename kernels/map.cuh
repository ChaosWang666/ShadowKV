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
 * @file map.cuh
 * @brief ShadowKV 稀疏注意力的高效哈希映射工具
 * 
 * 本文件实现了 ShadowKV 稀疏注意力机制中的核心映射操作，主要功能包括：
 * 
 * 1. **高效哈希映射**：基于开放寻址法的线程安全哈希表
 * 2. **稀疏索引管理**：将稀疏位置映射到连续索引空间
 * 3. **并行插入操作**：支持多线程并发的键值对插入
 * 4. **快速查找机制**：优化的哈希函数和冲突解决策略
 * 
 * 核心设计思想：
 * - 使用共享内存实现高速哈希表，减少全局内存访问
 * - 采用线性探测法解决哈希冲突，保证访问的局部性
 * - 基于原子操作实现线程安全的并发插入
 * - 针对 GPU 架构优化的哈希函数设计
 * 
 * 应用场景：
 * - 稀疏注意力中的位置索引映射
 * - KV 缓存的稀疏访问模式优化
 * - 动态序列长度的索引重排
 * - 批处理中的个性化稀疏模式处理
 * 
 * 性能优化策略：
 * - 2 的幂次表大小，使用位掩码代替模运算
 * - 循环展开减少分支开销
 * - 内存合并访问模式优化
 * - 基于线程块的协作式初始化
 */

#ifndef MAP_CUH
#define MAP_CUH

#include <limits.h>

/**
 * 哈希表配置参数
 * 
 * 这些宏定义控制哈希映射的性能和行为特性：
 */

/**
 * 空键标识符
 * 
 * 使用 -1 作为哈希表中空槽位的标识，因为：
 * - 在稀疏注意力中，位置索引通常为非负数
 * - -1 不会与有效的位置索引冲突
 * - 便于原子比较和交换操作的实现
 */
#define EMPTY_KEY -1

#ifndef TABLE_SIZE
/**
 * 哈希表大小配置
 * 
 * 默认值 1024 的选择考虑：
 * - 2 的幂次：支持高效的位掩码哈希（避免昂贵的模运算）
 * - 大于线程块大小：确保每个线程都有足够的槽位空间
 * - 平衡内存使用：1024 * 8 字节 = 8KB 共享内存（合理范围）
 * - 冲突概率：在典型稀疏度下提供良好的负载因子
 * 
 * 硬件适配建议：
 * - A100/H100: 可增大到 2048（充足的共享内存）
 * - V100/RTX 系列: 1024（平衡性能和资源）
 * - 较老架构: 512-1024（受限于共享内存）
 */
#define TABLE_SIZE 1024 // Chosen as a power of 2 greater than 256 for efficiency
#endif

#ifndef SORT_OFFSET
/**
 * 偏移排序控制标志
 * 
 * 控制是否对稀疏偏移进行排序：
 * - 1: 启用排序，优化内存访问局部性
 * - 0: 禁用排序，保持原始访问顺序
 * 
 * 排序的权衡：
 * - 优势：提高缓存命中率，减少内存延迟
 * - 劣势：增加预处理开销，可能影响小规模操作
 */
#define SORT_OFFSET 1
#endif

#ifndef BLOCK_SIZE_MAP
/**
 * 映射操作的线程块大小
 * 
 * 针对哈希映射操作优化：
 * - 256 线程：最大化 SM 占用率和并行度
 * - 与哈希表大小的关系：TABLE_SIZE / BLOCK_SIZE_MAP = 4（每线程处理 4 个槽位）
 * - 适合共享内存的协作式操作模式
 * - 平衡计算和内存访问的工作负载
 */
#define BLOCK_SIZE_MAP 256
#endif

/**
 * @brief 快速哈希函数
 * 
 * 针对 GPU 架构优化的高效哈希函数，专为稀疏注意力的位置索引设计。
 * 假设 TABLE_SIZE 是 2 的幂次，使用位掩码代替昂贵的模运算。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param key 输入键值（通常是位置索引）
 * @return 哈希值（0 到 LUT_SIZE-1 的范围内）
 * 
 * 设计原理：
 * - 简单位掩码：key & (LUT_SIZE - 1)
 *   优势：单指令完成，延迟极低
 *   适用：位置索引具有良好的分布特性
 * 
 * 复杂哈希（已禁用）：
 * - 多轮异或和乘法运算
 *   优势：更好的分布均匀性
 *   劣势：更高的计算开销
 * 
 * 性能考虑：
 * - GPU 上位运算比乘法更快
 * - 简单哈希在稀疏注意力场景下表现良好
 * - 避免分支和复杂运算，保持 warp 内线程同步
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ unsigned int fast_hash(int key)
{
    return key & (LUT_SIZE - 1); // 简单位掩码哈希计算

    // 复杂哈希函数（已禁用，保留用于特殊场景）
#if 0
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key & (LUT_SIZE - 1);
#endif
}

/**
 * @brief 向哈希映射中插入键值对
 * 
 * 使用开放寻址法（线性探测）实现线程安全的键值对插入操作。
 * 支持多线程并发插入，是稀疏注意力索引映射的核心操作。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param key 要插入的键（通常是稀疏位置索引）
 * @param value 对应的值（通常是连续索引或线程 ID）
 * @param map_keys 哈希表键数组指针
 * @param map_values 哈希表值数组指针
 * 
 * 算法流程：
 * 1. 计算初始哈希位置
 * 2. 使用原子比较交换尝试插入
 * 3. 如果冲突，线性探测到下一个位置
 * 4. 重复直到找到空槽位或相同键
 * 
 * 线程安全性：
 * - 使用 atomicCAS 保证插入的原子性
 * - 支持多线程并发插入不同的键
 * - 相同键的重复插入会更新值
 * 
 * 性能优化：
 * - 线性探测保证内存访问的局部性
 * - 位掩码代替模运算提高探测效率
 * - 无锁设计避免线程阻塞
 * 
 * 适用场景：
 * - 稀疏位置到连续索引的映射
 * - 动态构建访问模式的查找表
 * - 批处理中的个性化索引重排
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void insert_map(int key, int value, int *map_keys, int *map_values)
{
    unsigned int pos = fast_hash<MAP_SIZE, LUT_SIZE>(key);
    while (true)
    {
        // 原子比较交换：如果当前位置为空，则插入新键
        int existing_key = atomicCAS(&map_keys[pos], EMPTY_KEY, key);
        if (existing_key == EMPTY_KEY || existing_key == key)
        {
            // 成功插入或找到相同键，更新对应的值
            map_values[pos] = value;
            break;
        }
        // 线性探测：移动到下一个位置（使用位掩码实现环形查找）
        pos = (pos + 1) & (LUT_SIZE - 1);
    }
}

/**
 * @brief 重置哈希映射表
 * 
 * 清空哈希表中的所有键，将所有槽位标记为空。
 * 使用线程块协作的方式高效地初始化整个哈希表。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param map_keys 哈希表键数组指针
 * 
 * 算法特点：
 * - 每个线程负责多个槽位的清理
 * - 循环展开提高内存访问效率
 * - 步长为 MAP_SIZE，确保内存合并访问
 * 
 * 性能优化：
 * - 使用 #pragma unroll 完全展开循环
 * - 连续内存访问模式，最大化带宽利用率
 * - 避免分支和条件判断，保持 warp 同步
 * 
 * 使用场景：
 * - 新一轮稀疏注意力计算前的表初始化
 * - 批处理中不同样本间的表重用
 * - 动态序列长度变化时的表重置
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void reset_map(int *map_keys)
{
    int id = threadIdx.x;
#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        map_keys[id] = EMPTY_KEY;  // 将当前槽位标记为空
        id += MAP_SIZE;            // 移动到下一个负责的槽位
    }
}

/**
 * @brief 初始化哈希映射表
 * 
 * 基于每个线程的键值和线程 ID 初始化哈希表。
 * 这是稀疏注意力中构建位置索引映射的标准流程。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param key 每个线程要插入的键值（通常是稀疏位置索引）
 * @param map_keys 哈希表键数组指针
 * @param map_values 哈希表值数组指针
 * 
 * 算法流程：
 * 1. 重置整个哈希表（清空所有槽位）
 * 2. 线程块同步，确保重置完成
 * 3. 每个线程插入自己的键值对 (key, threadIdx.x)
 * 
 * 设计思想：
 * - 将稀疏位置索引映射到连续的线程 ID
 * - 线程 ID 作为值，便于后续的数据收集操作
 * - 支持动态的稀疏模式构建
 * 
 * 同步保证：
 * - __syncthreads() 确保所有线程完成重置后再插入
 * - 避免插入过程中的数据竞争
 * 
 * 应用场景：
 * - 稀疏注意力的位置索引重排
 * - KV 缓存的稀疏访问模式构建
 * - 动态序列长度的索引映射
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void init_map(int key, int *map_keys, int *map_values)
{
    reset_map<MAP_SIZE, LUT_SIZE>(map_keys);                              // 清空哈希表
    __syncthreads();                                                       // 确保所有线程完成重置
    insert_map<MAP_SIZE, LUT_SIZE>(key, threadIdx.x, map_keys, map_values); // 插入 (key, threadIdx.x) 键值对
}

/**
 * @brief 将哈希映射从共享内存写回全局内存
 * 
 * 高效地将共享内存中的哈希表数据传输到全局内存。
 * 用于保存计算结果或在不同内核间传递映射信息。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param s_keys 共享内存中的键数组指针
 * @param s_values 共享内存中的值数组指针
 * @param g_keys 全局内存中的键数组指针
 * @param g_values 全局内存中的值数组指针
 * 
 * 内存布局：
 * - 每个线程块在全局内存中有独立的 LUT_SIZE 大小区域
 * - 偏移计算：blockIdx.x * LUT_SIZE + 线程内偏移
 * - 保证不同线程块间的数据隔离
 * 
 * 性能优化：
 * - 循环展开减少控制开销
 * - 合并内存访问模式，最大化带宽
 * - 每个线程处理多个元素，提高并行度
 * 
 * 使用场景：
 * - 跨内核的映射信息传递
 * - 稀疏模式的持久化存储
 * - 多阶段计算的中间结果保存
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void write_back_map(int *s_keys, int *s_values, int *g_keys, int *g_values)
{
    int id = threadIdx.x;                                    // 共享内存中的索引
    int offset = blockIdx.x * LUT_SIZE + threadIdx.x;       // 全局内存中的起始偏移
#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        g_keys[offset] = s_keys[id];      // 写回键
        g_values[offset] = s_values[id];  // 写回值
        id += MAP_SIZE;                   // 移动到下一个共享内存位置
        offset += MAP_SIZE;               // 移动到下一个全局内存位置
    }
}

/**
 * @brief 从全局内存加载哈希映射到共享内存
 * 
 * 高效地将全局内存中的哈希表数据加载到共享内存。
 * 用于在内核开始时恢复之前保存的映射信息。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param s_keys 共享内存中的键数组指针
 * @param s_values 共享内存中的值数组指针
 * @param g_keys 全局内存中的键数组指针
 * @param g_values 全局内存中的值数组指针
 * 
 * 内存访问模式：
 * - 从全局内存的连续区域读取数据
 * - 写入共享内存的对应位置
 * - 保持与 write_back_map 相同的内存布局
 * 
 * 性能优化：
 * - 合并全局内存访问，最大化带宽利用率
 * - 循环展开减少控制流开销
 * - 并行加载提高数据传输效率
 * 
 * 使用场景：
 * - 多阶段计算的状态恢复
 * - 跨内核的映射信息传递
 * - 预计算映射的快速加载
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ void load_map(int *s_keys, int *s_values, int *g_keys, int *g_values)
{
    int id = threadIdx.x;                                    // 共享内存中的索引
    int offset = blockIdx.x * LUT_SIZE + threadIdx.x;       // 全局内存中的起始偏移

#pragma unroll
    for (int i = 0; i < LUT_SIZE; i += MAP_SIZE)
    {
        s_keys[id] = g_keys[offset];      // 加载键到共享内存
        s_values[id] = g_values[offset];  // 加载值到共享内存
        id += MAP_SIZE;                   // 移动到下一个共享内存位置
        offset += MAP_SIZE;               // 移动到下一个全局内存位置
    }
}

/**
 * @brief 在哈希映射中查找键对应的值
 * 
 * 使用线性探测法在哈希表中查找指定键的对应值。
 * 这是稀疏注意力中索引映射查询的核心操作。
 * 
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * @tparam LUT_SIZE 查找表大小（默认为 TABLE_SIZE）
 * @param key 要查找的键值
 * @param map_keys 哈希表键数组指针
 * @param map_values 哈希表值数组指针
 * @return 找到的值，如果键不存在则返回 -1
 * 
 * 算法流程：
 * 1. 计算键的初始哈希位置
 * 2. 检查当前位置的键是否匹配
 * 3. 如果匹配，返回对应的值
 * 4. 如果是空槽位，说明键不存在，返回 -1
 * 5. 否则线性探测到下一个位置，重复步骤 2-4
 * 
 * 性能特点：
 * - 平均时间复杂度：O(1)
 * - 最坏时间复杂度：O(n)（表满时）
 * - 内存访问局部性好，缓存友好
 * 
 * 返回值约定：
 * - 成功：返回键对应的值（通常是线程 ID 或连续索引）
 * - 失败：返回 -1，表示键在映射中不存在
 * 
 * 使用场景：
 * - 稀疏位置到连续索引的转换
 * - 检查某个位置是否在稀疏模式中
 * - 获取位置对应的处理线程 ID
 */
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__device__ int lookup_map(int key, int *map_keys, int *map_values)
{
    unsigned int pos = fast_hash<MAP_SIZE, LUT_SIZE>(key);  // 计算初始哈希位置

    while (true)
    {
        if (map_keys[pos] == key)
        {
            // 找到匹配的键，返回对应的值
            return map_values[pos];
        }
        if (map_keys[pos] == EMPTY_KEY)
        {
            // 遇到空槽位，说明键不存在
            return -1; // Key not found
        }
        // 线性探测：移动到下一个位置
        pos = (pos + 1) & (LUT_SIZE - 1);
    }
}

// debug only
__device__ void warp_bitonic_sort(int &key, int &value) {
   // Bitonic sort within a warp
    int lane = threadIdx.x & 31;
    for ( int k = 2; k <= 32; k = 2 * k) {
        for ( int j = k >> 1; j > 0; j = j >> 1) {
            bool dir = (( lane & j) != 0) ^ (( lane & k) != 0) ;
            int partnerKey = __shfl_xor_sync(0xFFFFFFF , key , j);
            int partnerValue = __shfl_xor_sync(0xFFFFFFFF, value, j);

            if((key > partnerKey) ^ dir) {
              key = partnerKey;
              value = partnerValue;
            } 
         }
    }
}

// bitonic sort within a warp using shuffle instructions
__device__ void warp_bitonic_sort2(int &key, int &value, int &key2, int &value2) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 2; k <= 32; k = 2 * k) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bool dir = (( lane & j) != 0) ^ (( lane & k) != 0) ;
            int partnerKey = __shfl_xor_sync(0xFFFFFFF , key , j);
            int partnerValue = __shfl_xor_sync(0xFFFFFFFF, value, j);

            int partnerKey2 = __shfl_xor_sync(0xFFFFFFF , key2 , j);
            int partnerValue2 = __shfl_xor_sync(0xFFFFFFFF, value2, j);


            if((key > partnerKey) ^ dir) {
              key = partnerKey;
              value = partnerValue;
            } 

            if((key2 > partnerKey2) ^ dir) {
              key2 = partnerKey2;
              value2 = partnerValue2;
            } 
         }
    }
}

template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ void merge_sort(int *keys, int *values) {
    int localIdx = threadIdx.x;
    #pragma unroll
    for (int size = 2; size <= MAP_SIZE; size *= 2) {
        #pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int ixj = localIdx ^ stride;

            if (ixj > localIdx) {
                // Determine the direction of sorting (ascending or descending)
                bool ascending = ((localIdx & size) == 0);

                // Compare and possibly swap
                int key1 = keys[localIdx];
                int key2 = keys[ixj];

                if ((key1 > key2) == ascending) {
                    // Swap elements
                    keys[localIdx] = key2;
                    keys[ixj] = key1;

                    int temp = values[localIdx] ;
                    values[localIdx]  = values[ixj];
                    values[ixj] = temp;
                }
            }
            __syncthreads(); // Synchronize threads within the block
        }
    }
}

// block-level sort 
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ void block_sort2(int *keys, int *values, int mp) {
    int tid = threadIdx.x;

    int key1 = tid < mp ? keys[tid] : INT_MAX;
    int value1 = values[tid];
    int key2 = tid < mp ? -1 : keys[tid];
    int value2 = values[tid];

    //warp_bitonic_sort2(key1, value1, key2, value2);
    //__syncthreads();

    keys[tid] = key1;
    values[tid] = value1;
    __syncthreads();
    merge_sort<MAP_SIZE>(keys, values);
     // hidden sync

    key1 = keys[tid];
    value1 = values[tid];
    // no need sync 
    keys[tid] = key2;
    values[tid] = value2;

    __syncthreads();
    merge_sort<MAP_SIZE>(keys, values);
    // hidden sync

    keys[tid] = tid < mp ? key1 : keys[tid];
    values[tid] = tid < mp ? value1 : values[tid];
    __syncthreads();
}

// given hits and keys of a thread-block
// reorder keys, and values
// all values with true hits are packed together.
// then value with false hits are assigned by counts (from 0, 1, 2,...) and packed together.
// and return how many true hits
// e.g. hits:      [false, true, false, true],
//      keys:      [23,    40,   52,    99],
//      values:    [345,   455,  544,   24],
//      reorders:  [40,    99,   23,    52],
//      new_values:[455,   24,    0,    1],
//      return: 2
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ int update_keys_and_offsets(bool hit, int key, int value, int *reorder_key, int *new_values, int *warp_sums)
{
    int thid = threadIdx.x;
    int laneId = thid % 32; // Warp lane index
    int warpId = thid / 32; // Warp index within the block

    if (warpId == 0 && thid < 32)
    {
        warp_sums[thid] = 0;
    }
    __syncthreads();

    // Step 1: Compute the ballot mask for the warp
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    unsigned int ballot2 = __ballot_sync(0xFFFFFFFF, !hit);
    // Step 2: Compute the exclusive prefix sum within the warp
    unsigned int prefix = __popc(ballot & ((1U << laneId) - 1));
    unsigned int prefix2 = __popc(ballot2 & ((1U << laneId) - 1));

    unsigned int result = prefix;
    unsigned int result2 = prefix2;
    if (laneId == 31)
    {
        warp_sums[warpId] = prefix + hit;
        warp_sums[MAP_SIZE / 32 + warpId] = prefix2 + !hit;
    }
    __syncthreads();

    // Perform a warp-level scan to propagate the sum across warps
    if (warpId == 0 && thid < 32)
    {
        int warp_sum = warp_sums[thid];

#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            int n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (laneId >= offset)
                warp_sum += n;
        }
        warp_sums[thid] = warp_sum;
    }
    __syncthreads();

    if (warpId > 0)
    {
        result += warp_sums[warpId - 1];
    }
    result2 += warp_sums[MAP_SIZE / 32 + warpId - 1];
    int idx = hit ? result : result2;
    int value_new = hit ? value : result2 - warp_sums[MAP_SIZE / 32 - 1];
    new_values[idx] = value_new;
    reorder_key[idx] = key;
    __syncthreads();
    return warp_sums[MAP_SIZE / 32 - 1];
}

// given hits and keys of a thread-block
// reorder keys, and values
// all values with true hits are packed together.
// then value with false hits are assigned by counts (from 0, 1, 2,...) and packed together.
// and return how many true hits
// e.g. hits:      [false, true, false, true],
//      keys:      [23,    40,   52,    99],
//      values:    [345,   455,  544,   24],
//      reorders:  [40,    99,   23,    52],
//      new_values:[455,   24,   23,    52],
//      return: 2
template<int MAP_SIZE = BLOCK_SIZE_MAP>
__device__ int update_keys_and_mixed_offsets(bool hit, int key, int value, int *reorder_key, int *new_values, int *warp_sums)
{
    int thid = threadIdx.x;
    int laneId = thid & 31; // Warp lane index
    int warpId = thid >> 5; // Warp index within the block

    if (thid < 64)
    {
        warp_sums[thid] = 0;
    }
    __syncthreads();

    // Step 1: Compute the ballot mask for the warp
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    unsigned int ballot2 = __ballot_sync(0xFFFFFFFF, !hit);
    // Step 2: Compute the exclusive prefix sum within the warp
    unsigned int prefix = __popc(ballot & ((1U << laneId) - 1));
    unsigned int prefix2 = __popc(ballot2 & ((1U << laneId) - 1));

    unsigned int result = prefix;
    unsigned int result2 = prefix2;
    if (laneId == 31)
    {
        warp_sums[warpId] = prefix + hit;
        warp_sums[MAP_SIZE / 32 + warpId] = prefix2 + !hit; // max id is 64
    }
    __syncthreads();

    // Perform a warp-level scan to propagate the sum across warps
    if (thid < 64)
    {
        int warp_sum = warp_sums[thid];

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            int n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (laneId >= offset)
                warp_sum += n;
        }
        warp_sums[thid] = warp_sum;
    }
    __syncthreads();

    if(warpId == 0) {
        warp_sums[32 + laneId] += warp_sums[31];
    }

    __syncthreads();

    if (warpId > 0)
    {
        result += warp_sums[warpId - 1];
    }
    result2 += warp_sums[MAP_SIZE / 32 + warpId - 1];
    int idx = hit ? result : result2;
    int value_new = hit ? value : key;
    new_values[idx] = value_new;
    reorder_key[idx] = key;
    __syncthreads();  
    return warp_sums[MAP_SIZE / 32 - 1];
}


// insert keys with values threadblock (not used in Shadow-KV)
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void map_insert(int *keys, int *map_keys, int *map_values)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];

    reset_map<MAP_SIZE, LUT_SIZE>(s_keys);
    __syncthreads();

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;
    int key = keys[offset];
    int value = threadIdx.x;

    insert_map<MAP_SIZE, LUT_SIZE>(key, value, s_keys, s_values);
    __syncthreads();

    // write back map
    write_back_map<MAP_SIZE, LUT_SIZE>(s_keys, s_values, map_keys, map_values);
}


// Not used in Shadow-KV
// create a shared memory map using orig_keys and threadIdx.x as values
// then look up values based on query_keys
// then reorder keys and update values
// Note: query_keys and g_reorder_keys can use the same pointer
template<int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void reorder_keys_and_offsets(
    int *orig_keys,
    int *query_keys,
    int *g_reorder_keys,
    int *g_offsets,
    int *g_hit_cnt)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];
    __shared__ int warp_sums[32]; // Assuming a maximum of 32 warps per block
    __shared__ int offsets[MAP_SIZE];
    __shared__ int reorder_keys[MAP_SIZE];

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    // create a shared memory map
    int old_key = orig_keys[offset];
    init_map<MAP_SIZE, LUT_SIZE>(old_key, s_keys, s_values);
    __syncthreads();

    // creat a new key as value (tid) for lookup
    int new_key = query_keys[offset];
    int value = lookup_map<MAP_SIZE, LUT_SIZE>(new_key, s_keys, s_values);
    bool hit = value != EMPTY_KEY;

    int hit_cnt = update_keys_and_offsets<MAP_SIZE>(hit, new_key, value, reorder_keys, offsets, warp_sums);
    // a hidden sync above
    // write out
    g_reorder_keys[offset] = reorder_keys[threadIdx.x];
    g_offsets[offset] = offsets[threadIdx.x];
    if (threadIdx.x == 0)
    {
        g_hit_cnt[blockIdx.x] = hit_cnt;
    }
}

// create a shared memory map using orig_keys and threadIdx.x as values
// then look up values based on query_keys
// then reorder keys and update values
// Note: query_keys and g_reorder_keys can use the same pointer
template<typename T, int MAP_SIZE = BLOCK_SIZE_MAP, int LUT_SIZE = TABLE_SIZE>
__global__ void reorder_keys_and_mixed_offsets(
    T *orig_keys,
    T *query_keys,
    T *g_reorder_keys,
    int *g_offsets,
    int *g_hit_cnt)
{
    __shared__ int s_values[LUT_SIZE];
    __shared__ int s_keys[LUT_SIZE];
    __shared__ int warp_sums[64]; // Assuming a maximum of 32 warps per block
    __shared__ int offsets[MAP_SIZE];
    __shared__ int reorder_keys[MAP_SIZE];

    int offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    // create a shared memory map
    int old_key = (T) (orig_keys[offset]);  // might be a cast
    init_map<MAP_SIZE, LUT_SIZE>(old_key, s_keys, s_values);
    __syncthreads();

    // creat a new key as value (tid) for lookup
    int new_key = (T) (query_keys[offset]); // might be a cast

    int value = lookup_map<MAP_SIZE, LUT_SIZE>(new_key, s_keys, s_values);
    bool hit = value != EMPTY_KEY;

    int hit_cnt = update_keys_and_mixed_offsets<MAP_SIZE>(hit, new_key, value, reorder_keys, offsets, warp_sums);

#if SORT_OFFSET
    //sort offset as keys and reorder_keys as values
    block_sort2<MAP_SIZE>(offsets, reorder_keys, hit_cnt);
#endif

    // a hidden sync above
    // write out
    g_reorder_keys[offset] = (T) (reorder_keys[threadIdx.x]); // might be a cast
    g_offsets[offset] = offsets[threadIdx.x];
    if (threadIdx.x == 0)
    {
        g_hit_cnt[blockIdx.x] = hit_cnt;
    }
}

#endif // MAP_CUH
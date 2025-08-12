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
 * @file copy.cuh
 * @brief 稀疏注意力的高效内存拷贝工具
 * 
 * 本文件实现了 ShadowKV 稀疏注意力机制中的核心内存操作，主要功能包括：
 * 
 * 1. **稀疏 Gather 拷贝**：根据稀疏索引高效收集分散的数据
 * 2. **批量数据传输**：优化的批量内存拷贝操作
 * 3. **线程同步机制**：基于内存屏障的高效同步原语
 * 4. **向量化访问**：利用向量化指令提高内存带宽利用率
 * 
 * 核心设计思想：
 * - 使用共享内存作为缓冲区，减少全局内存访问
 * - 通过 coalesced 访问模式优化内存带宽
 * - 实现细粒度的线程同步，避免不必要的全局同步
 * - 支持不同数据类型的模板化实现
 * 
 * 性能优化策略：
 * - 循环展开减少分支开销
 * - 向量化内存访问提高带宽利用率
 * - 异步拷贝与计算重叠
 * - 基于 acquire-release 语义的轻量级同步
 */

#ifndef COPY_CUH
#define COPY_CUH

/**
 * 拷贝操作的配置参数
 * 
 * 这些宏定义控制内存拷贝的性能和资源使用：
 */

#ifndef CPY_SIZE
/**
 * 每次拷贝的批量大小配置
 * 
 * 这个参数控制稀疏 gather 操作的批处理大小，直接影响：
 * - 共享内存使用量：CPY_SIZE * sizeof(T) * threads_per_block
 * - 内存合并访问效率：较大的批量有利于内存合并
 * - 寄存器压力：过大会增加寄存器使用
 * 
 * 硬件建议：
 * - A100/H100: 64+ (充足的共享内存和寄存器)
 * - V100/RTX 系列: 32 (平衡性能和资源)
 * - 较老架构: 16-32 (受限于共享内存)
 */
#define CPY_SIZE 32
#endif

#ifndef UNROLL_FACTOR
/**
 * 循环展开因子配置
 * 
 * 控制内层循环的展开程度，影响：
 * - 指令级并行性：更多展开提供更多并行机会
 * - 代码大小：过度展开会增加指令缓存压力
 * - 编译时间：展开因子过大会显著增加编译时间
 * 
 * 推荐值：4-16，根据具体内核复杂度调整
 */
#define UNROLL_FACTOR 8
#endif

/**
 * 循环展开策略定义
 * 
 * 采用分层展开策略：
 * - 外层循环：最小展开 (#pragma unroll 1)
 *   保持灵活性，适应不同的数据大小
 * - 内层循环：完全展开 (#pragma unroll UNROLL_FACTOR)
 *   最大化指令级并行性和内存带宽利用率
 */
#define COPY_UNROLL_OUTER #pragma unroll 1
#define COPY_UNROLL_INNER #pragma unroll UNROLL_FACTOR

#ifndef BLOCK_SIZE_CP
/**
 * 拷贝操作的线程块大小
 * 
 * 针对内存密集型操作优化：
 * - 128 线程：平衡内存带宽和占用率
 * - 适合 gather/scatter 操作的不规则内存访问模式
 * - 与 warp 大小 (32) 的倍数关系确保高效调度
 */
#define BLOCK_SIZE_CP 128
#endif

#ifndef BLOCK_SIZE_MAP
/**
 * 映射操作的线程块大小
 * 
 * 针对计算密集型操作优化：
 * - 256 线程：最大化 SM 占用率
 * - 适合规则的索引计算和映射操作
 * - 更大的线程块有利于摊销启动开销
 */
#define BLOCK_SIZE_MAP 256
#endif

#ifndef SORT_OFFSET
/**
 * 排序偏移控制标志
 * 
 * 控制稀疏索引的排序行为：
 * - 1: 启用索引排序，优化内存访问局部性
 * - 0: 禁用排序，保持原始索引顺序
 * 
 * 排序的权衡：
 * - 优势：提高缓存命中率，减少内存延迟
 * - 劣势：增加预处理开销，可能影响小规模操作
 */
#define SORT_OFFSET 1
#endif

namespace
{
    /**
     * 基于 CUTLASS 的同步原语实现（经过修改优化）
     * 
     * 这些函数实现了高效的线程间同步机制，用于协调稀疏拷贝操作。
     */

    /**
     * @brief 原子递增操作（带释放语义）
     * 
     * 实现线程安全的计数器递增，用于同步多个线程块的进度。
     * 使用 acquire-release 内存模型确保内存操作的可见性。
     * 
     * @param ptr 指向计数器的指针
     * 
     * 架构优化：
     * - Volta+ (SM 7.0+): 使用硬件内存屏障和原子指令
     * - 较老架构: 使用软件内存屏障和传统原子操作
     */
    __forceinline__ __device__ void red_release(unsigned int *ptr)
    {
#if (__CUDA_ARCH__ >= 700)
        // Volta+ 架构：使用 acquire-release 语义的内存屏障
        asm volatile("fence.acq_rel.gpu;\n");
        // 使用 relaxed 语义的原子递增（已有屏障保证顺序）
        asm volatile("red.relaxed.gpu.global.inc.u32 [%0], 1;\n" : : "l"(ptr));
#else
        // 较老架构：使用传统的内存屏障和原子操作
        __threadfence();
        atomicInc(ptr, 1);
#endif // (__CUDA_ARCH__ >= 700)
    }

    /**
     * @brief 线程块到达并递增计数器
     * 
     * 当线程块完成某个阶段的工作时调用，用于通知其他线程块。
     * 只有线程 0 执行实际的原子操作，减少竞争。
     * 
     * @param ptr 指向全局计数器的指针
     */
    __forceinline__ __device__ void arrive_inc(unsigned int *ptr)
    {
        __syncthreads();  // 确保线程块内所有线程都完成工作
        if (threadIdx.x == 0)
        {
            red_release(ptr);  // 只有线程 0 执行原子递增
        }
    }

    /**
     * @brief 原子加载操作（带获取语义）
     * 
     * 读取全局计数器的当前值，确保能看到其他线程的写入。
     * 使用 acquire 语义保证内存操作的可见性。
     * 
     * @param ptr 指向计数器的指针
     * @return 计数器的当前值
     */
    __forceinline__ __device__ int ld_acquire(unsigned int *ptr)
    {
        int state = 0;
#if (__CUDA_ARCH__ >= 700)
        // Volta+ 架构：使用 acquire 语义的加载指令
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#else
        // 较老架构：使用缓存一致性加载
        asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif // (__CUDA_ARCH__ >= 700)
        return state;
    }

    /**
     * @brief 等待计数器达到指定值
     * 
     * 实现线程块间的同步等待，直到全局计数器达到期望值。
     * 使用自旋锁机制，适合短时间等待的场景。
     * 
     * @param ptr 指向计数器的指针
     * @param val 期望的计数器值（默认为 0）
     * 
     * 优化策略：
     * - 只有线程 0 执行轮询，减少内存带宽消耗
     * - 最小循环展开，避免过度占用指令缓存
     * - 使用 acquire 语义确保能及时看到更新
     */
    __forceinline__ __device__ void wait_eq(unsigned int *ptr, unsigned int val = 0)
    {
        if (threadIdx.x == 0)
        {
            // 自旋等待：最小展开以保持响应性
#pragma unroll 1
            while (ld_acquire(ptr) != val)
            {
                // 空循环体：持续轮询直到条件满足
            }
        }
        __syncthreads();  // 确保所有线程都知道等待结束
    }

    /**
     * @brief 等待计数器达到指定值并重置
     * 
     * 类似于 wait_eq，但在等待完成后会将计数器重置为 0。
     * 用于实现可重复使用的同步屏障。
     * 
     * @param ptr 指向计数器的指针
     * @param val 期望的计数器值（默认为 0）
     * 
     * 使用场景：
     * - 多轮迭代的同步点
     * - 需要重置状态的屏障同步
     * - 避免计数器溢出的周期性重置
     */
    __forceinline__ __device__ void wait_eq_reset(unsigned int *ptr, unsigned int val = 0)
    {

        if (threadIdx.x == 0)
        {
// Spin-loop
#pragma unroll 1
            while (atomicCAS(ptr, val, 0) != val)
            {
                //  printf("ptr %d\n", ptr[0]);
            }
        }
        __syncthreads();
    }

}

/**
 * @brief 基于稀疏索引的 Gather 拷贝操作
 * 
 * 根据稀疏索引数组从源数据中收集元素，是稀疏注意力的核心操作。
 * 使用共享内存作为缓冲区，优化内存访问模式。
 * 
 * @tparam T 数据类型（支持任意 POD 类型）
 * @param src 源数组指针
 * @param dst 目标数组指针
 * @param s_data 共享内存缓冲区
 * @param s_offsets 稀疏索引数组（指定要收集的元素位置）
 * @param src_buffer_offset 源缓冲区偏移
 * @param src_buffer_size 源缓冲区大小（每个块）
 * @param dst_buffer_offset 目标缓冲区偏移
 * @param dst_buffer_size 目标缓冲区大小（每个块）
 * @param start 起始索引
 * @param end 结束索引
 * @param dst_start 目标起始位置
 * @param bid 块 ID
 * 
 * 算法流程：
 * 1. 分批处理：每批最多 CPY_SIZE 个元素
 * 2. 读取阶段：根据索引从源数组读取到共享内存
 * 3. 写入阶段：从共享内存写入到目标数组
 * 4. 同步保证：确保读写操作的正确性
 * 
 * 性能优化：
 * - 使用共享内存减少全局内存访问延迟
 * - 批量处理提高内存带宽利用率
 * - 线程协作实现高效的数据移动
 * 
 * 线程配置：
 * - 假设每个线程块有 128 或 256 个线程
 * - 共享内存使用：2KB = 128 threads x sizeof(int4) 或 256 threads x sizeof(int2)
 * - 每个线程加载 16 或 8 字节
 */
template <typename T>
__device__ void gather_copy(
    T *src,
    T *dst,
    T *s_data,
    int *s_offsets,      // gather offsets
    int src_buffer_offset,
    int src_buffer_size, // per block
    int dst_buffer_offset,
    int dst_buffer_size, // per block
    int start,
    int end,
    int dst_start,
    int bid)
{

    int64_t bid_64 = bid; // ned to cast to int64_t to make sure  bid_64 * src_buffer_size not overflow.
    // assume using src_buffer_size * sizeof(bf16) Byte / sizeof(T) Byte
    int64_t src_base = (bid_64 * src_buffer_size + src_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_base = (bid_64 * dst_buffer_size + dst_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_offset = dst_base + dst_start * BLOCK_SIZE_CP + threadIdx.x;           // int64_t to avoid overflow

    int64_t offset_index = start;
    int64_t offset_end = end;

    COPY_UNROLL_OUTER
    while (offset_index < offset_end)
    {

        // read
        int64_t iter = 0;
        int64_t l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while ((iter < CPY_SIZE) && (offset_index < offset_end))
        {
            iter++;
            int64_t offset = s_offsets[offset_index]; // int64_t to avoid overflow
            offset_index++;
            int64_t src_offset = src_base + offset * BLOCK_SIZE_CP + threadIdx.x; // int64_t to avoid overflow
            s_data[l_offset] = src[src_offset];
            l_offset += BLOCK_SIZE_CP;
        }

        // write
        int64_t max_iter = iter;
        iter = 0;
        l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < max_iter)
        {
            iter++;
            dst[dst_offset] = s_data[l_offset];
            l_offset += BLOCK_SIZE_CP;
            dst_offset += BLOCK_SIZE_CP;
        }
        __syncthreads(); // must, avoid data racing
    }
}

/**
 * @brief 标准的批量内存拷贝操作
 * 
 * 实现高效的连续内存拷贝，使用共享内存作为中转缓冲区。
 * 与 gather_copy 不同，这是连续地址的直接拷贝。
 * 
 * @tparam T 数据类型（支持任意 POD 类型）
 * @param src 源数组指针
 * @param dst 目标数组指针
 * @param s_data 共享内存缓冲区
 * @param src_buffer_offset 源缓冲区偏移
 * @param src_buffer_size 源缓冲区大小（每个块）
 * @param dst_buffer_offset 目标缓冲区偏移
 * @param dst_buffer_size 目标缓冲区大小（每个块）
 * @param start 起始索引
 * @param end 结束索引
 * @param dst_start 目标起始位置
 * @param bid 块 ID（默认为 blockIdx.x）
 * 
 * 算法特点：
 * - 连续内存访问：优化内存带宽利用率
 * - 批量处理：减少同步开销
 * - 共享内存缓冲：隐藏全局内存延迟
 * 
 * 适用场景：
 * - 密集矩阵的块拷贝
 * - KV 缓存的连续更新
 * - 中间结果的批量传输
 * 
 * 线程配置：
 * - 假设每个线程块有 128 或 256 个线程
 * - 共享内存使用：2KB = 128 threads x sizeof(int4) 或 256 threads x sizeof(int2)
 * - 每个线程加载 16 或 8 字节
 */
template <typename T>
__device__ void copy(
    T *src,
    T *dst,
    T *s_data,
    int src_buffer_offset,
    int src_buffer_size, // per block
    int dst_buffer_offset,
    int dst_buffer_size, // per block
    int start,
    int end,
    int dst_start,
    int bid = blockIdx.x)
{
    int64_t bid_64 = bid;
    // assume using src_buffer_size * sizeof(bf16) Byte / sizeof(T) Byte
    int64_t src_base = (bid_64 * src_buffer_size + src_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t dst_base = (bid_64 * dst_buffer_size + dst_buffer_offset) * 2 / sizeof(T); // int64_t to avoid overflow
    int64_t src_offset = src_base + start * BLOCK_SIZE_CP + threadIdx.x;               // int64_t to avoid overflow
    int64_t dst_offset = dst_base + dst_start * BLOCK_SIZE_CP + threadIdx.x;           // int64_t to avoid overflow

    int64_t offset_index = start;
    int64_t offset_end = end;

    COPY_UNROLL_OUTER
    while (offset_index < offset_end)
    {
        // read
        int64_t iter = 0;
        int64_t l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < CPY_SIZE && offset_index < offset_end)
        {
            iter++;
            offset_index++;
            s_data[l_offset] = src[src_offset];
            src_offset += BLOCK_SIZE_CP;
            l_offset += BLOCK_SIZE_CP;
        }

        // here, not need __syncthreads

        // write
        int64_t max_iter = iter;
        iter = 0;
        l_offset = threadIdx.x;

        COPY_UNROLL_INNER
        while (iter < max_iter)
        {
            iter++;
            dst[dst_offset] = s_data[l_offset];
            l_offset += BLOCK_SIZE_CP;
            dst_offset += BLOCK_SIZE_CP;
        }
        __syncthreads(); // must, avoid data racing
    }
}

/**
 * @brief 固定起始和结束位置的 Gather 拷贝内核
 * 
 * 实现具有固定起始和结束索引的稀疏数据收集操作。
 * 适用于已知数据范围的批量稀疏拷贝场景。
 * 
 * @tparam T 数据类型
 * @tparam TKEYS 键类型（索引类型）
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param src 源数据指针
 * @param dst 目标数据指针
 * @param src_buffer_size 源缓冲区大小
 * @param dst_buffer_size 目标缓冲区大小
 * @param keys 稀疏索引键数组
 * @param start 固定起始位置
 * @param end 固定结束位置
 * 
 * 特点：
 * - 预定义的数据范围，无需动态计算边界
 * - 高效的批量处理，适合规则的稀疏模式
 * - 使用共享内存缓存索引，减少全局内存访问
 */
template <typename T, typename TKEYS, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gahter_copy_fixed_start_end(
    T *src, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    TKEYS *keys,
    int start,
    int end)
{

    extern __shared__ int s[];
    int *s_offsets = s; // MAP_SIZE
    T *s_data = (T *)(s + MAP_SIZE);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = keys[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }
    __syncthreads();

    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end, 0, blockIdx.x);
}

/**
 * @brief 可变起始位置和固定结束位置的 Gather 拷贝内核
 * 
 * 支持每个块具有不同起始位置但相同结束位置的稀疏拷贝。
 * 适用于不同长度序列但有统一截断点的场景。
 * 
 * @tparam T 数据类型
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param src 源数据指针
 * @param dst 目标数据指针
 * @param src_buffer_size 源缓冲区大小
 * @param dst_buffer_size 目标缓冲区大小
 * @param offsets 稀疏偏移数组
 * @param start_cnts 每个块的起始计数数组
 * @param end 固定结束位置
 * 
 * 应用场景：
 * - 变长序列的注意力计算
 * - 动态稀疏模式的数据收集
 * - 批处理中不同样本的个性化处理
 */
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_var_start_fixed_end(
    T *src, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    int *offsets,
    int *start_cnts,
    int end)
{

    extern __shared__ int s[];
    int *s_offsets = s;                  // BLOCK_SIZE_MAP
    int *start_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(start_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        start_cnt[0] = start_cnts[blockIdx.x];
    __syncthreads();

    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start_cnt[0], end, start_cnt[0], blockIdx.x);
}

/**
 * @brief 固定起始和可变结束位置的 Gather 拷贝内核（使用临时缓冲区）
 * 
 * 支持固定起始位置但每个块有不同结束位置的稀疏拷贝。
 * 使用临时缓冲区优化内存访问模式，避免数据竞争。
 * 
 * @tparam T 数据类型
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param src 源数据指针
 * @param temp 临时缓冲区指针
 * @param dst 目标数据指针
 * @param src_buffer_size 源缓冲区大小
 * @param dst_buffer_size 目标缓冲区大小
 * @param offsets 稀疏偏移数组
 * @param start 固定起始位置
 * @param end_cnts 每个块的结束计数数组
 * 
 * 优化策略：
 * - 使用临时缓冲区实现两阶段拷贝
 * - 根据 SORT_OFFSET 标志选择直接或间接拷贝
 * - 减少内存访问冲突，提高并发性能
 */
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gahter_copy_fixed_start_var_end_with_temp(
    T *src, T *temp, T *dst,
    int src_buffer_size,
    int dst_buffer_size,
    int *offsets,
    int start,
    int *end_cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                // BLOCK_SIZE_MAP
    int *end_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(end_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        end_cnt[0] = end_cnts[blockIdx.x];
    __syncthreads();

#if SORT_OFFSET
    gather_copy(src, dst, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start, blockIdx.x);
#else
    gather_copy(src, temp, s_data, s_offsets, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start, blockIdx.x);
    copy(temp, dst, s_data, 0, src_buffer_size, 0, dst_buffer_size, start, end_cnt[0], start);
#endif
}

/**
 * @brief 设备到设备的 Gather 拷贝内核
 * 
 * 专门用于 GPU 设备内存之间的稀疏数据拷贝操作。
 * 支持复杂的缓冲区布局和步长访问模式。
 * 
 * @tparam T 数据类型
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param d_values 设备数据指针（既是源也是目标）
 * @param temp 临时缓冲区指针（可选）
 * @param d_buffer_size 设备缓冲区大小
 * @param d_buffer_offset 设备缓冲区偏移
 * @param d_buffer_stride 设备缓冲区步长
 * @param offsets 稀疏偏移数组
 * @param start 起始位置
 * @param end_cnts 每个块的结束计数数组
 * 
 * 特殊功能：
 * - 支持原地操作（in-place）和临时缓冲区操作
 * - 处理复杂的内存布局（偏移和步长）
 * - 优化设备内存带宽利用率
 */
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_d2d(
    T *d_values, 
    T *temp, // 可选的临时缓冲区
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int start,
    int *end_cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                // BLOCK_SIZE_MAP
    int *end_cnt = s + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(end_cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        end_cnt[0] = end_cnts[blockIdx.x];
    __syncthreads();

#if SORT_OFFSET
    gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, start, end_cnt[0], start, blockIdx.x);
#else
    gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, start, end_cnt[0], start, blockIdx.x);
    copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, start, end_cnt[0], start);
#endif
}

/**
 * @brief 可变中点的 Gather 拷贝内核
 * 
 * 支持动态中点分割的稀疏拷贝操作，可以处理主机和设备内存之间的数据传输。
 * 根据中点将数据分为两部分进行不同的处理策略。
 * 
 * @tparam T 数据类型
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param h_values 主机数据指针
 * @param temp 临时缓冲区指针
 * @param d_values 设备数据指针
 * @param h_buffer_size 主机缓冲区大小
 * @param d_buffer_size 设备缓冲区大小
 * @param d_buffer_offset 设备缓冲区偏移
 * @param d_buffer_stride 设备缓冲区步长
 * @param offsets 稀疏偏移数组
 * @param cnts 每个块的中点计数数组
 * 
 * 算法逻辑：
 * - 前半部分：设备到设备的拷贝（0 到 cnt[0]）
 * - 后半部分：主机到设备的拷贝（cnt[0] 到 MAP_SIZE）
 * - 支持混合内存层次的高效数据管理
 */
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP>
__global__ void gather_copy_var_midpoint(
    T *h_values, T *temp, T *d_values,
    int h_buffer_size,
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int *cnts)
{

    extern __shared__ int s[];
    int *s_offsets = s;                    // BLOCK_SIZE_MAP
    int *cnt = s_offsets + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(cnt + sizeof(T) / 4);

    int key_offset = blockIdx.x * MAP_SIZE + threadIdx.x;
    int idx = threadIdx.x;

#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
    {
        cnt[0] = cnts[blockIdx.x];
    }
    __syncthreads();
#if SORT_OFFSET
    gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, blockIdx.x);
    gather_copy(h_values, d_values, s_data, s_offsets, 0, h_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], blockIdx.x);
#else
    gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, 0, cnt[0], 0, blockIdx.x);
    copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, blockIdx.x);
    gather_copy(h_values, d_values, s_data, s_offsets, 0, h_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], blockIdx.x);
#endif
}

/**
 * @brief 块特化的可变中点 Gather 拷贝内核
 * 
 * 使用块特化（Block Specialization）技术的高级稀疏拷贝内核。
 * 不同的线程块执行不同的任务，通过信号量进行精确同步。
 * 
 * @tparam T 数据类型
 * @tparam MAP_SIZE 映射大小（默认为 BLOCK_SIZE_MAP）
 * 
 * @param h_values 主机数据指针
 * @param temp 临时缓冲区指针
 * @param d_values 设备数据指针
 * @param h_buffer_size 主机缓冲区大小
 * @param d_buffer_size 设备缓冲区大小
 * @param d_buffer_offset 设备缓冲区偏移
 * @param d_buffer_stride 设备缓冲区步长
 * @param offsets 稀疏偏移数组
 * @param cnts 每个块的中点计数数组
 * @param signals 线程块间同步信号量
 * 
 * 块特化策略：
 * - 奇数块 (bs_task=1)：处理设备到设备的拷贝
 * - 偶数块 (bs_task=0)：处理主机到设备的拷贝
 * - 使用信号量实现精确的生产者-消费者同步
 * 
 * 性能优势：
 * - 并行执行不同类型的内存操作
 * - 减少全局同步开销
 * - 提高内存带宽利用率
 */
template <typename T, int MAP_SIZE = BLOCK_SIZE_MAP> 
__global__ void gather_copy_var_midpoint_BP(
    T *h_values, T *temp, T *d_values,
    int h_buffer_size,
    int d_buffer_size,
    int d_buffer_offset,
    int d_buffer_stride,
    int *offsets,
    int *cnts,
    unsigned int *signals)
{

    extern __shared__ int s[];
    int *s_offsets = s;                    // BLOCK_SIZE_MAP
    int *cnt = s_offsets + MAP_SIZE; // 1, but occupy 2 to avoid alignment issue
    T *s_data = (T *)(cnt + sizeof(T) / 4);

    int bid = blockIdx.x >> 1;
    int bs_task = blockIdx.x & 1; // 0 or 1

    int key_offset = bid * MAP_SIZE + threadIdx.x;

    int idx = threadIdx.x;
#pragma unroll
    for (int i = 0; i < MAP_SIZE / BLOCK_SIZE_CP; i++)
    {
        s_offsets[idx] = offsets[key_offset];
        idx += BLOCK_SIZE_CP;
        key_offset += BLOCK_SIZE_CP;
    }

    if (threadIdx.x == 0)
        cnt[0] = cnts[bid];
    __syncthreads();

    signals += bid;

    if (bs_task)
    {
#if SORT_OFFSET
        // d2d: from d_values to d_values
        gather_copy(d_values, d_values, s_data, s_offsets, d_buffer_offset, d_buffer_stride, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, bid);
        arrive_inc(signals);
#else 
        // d2d: to temp
        gather_copy(d_values, temp, s_data, s_offsets, d_buffer_offset, d_buffer_stride, 0, d_buffer_size, 0, cnt[0], 0, bid);
        arrive_inc(signals);
        wait_eq(signals);
        // d2d: from temp
        copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, 0, cnt[0], 0, bid);
#endif
    }
    else
    {
        // h2d: to temp
        gather_copy(h_values, temp, s_data, s_offsets, 0, h_buffer_size, 0, d_buffer_size, cnt[0], MAP_SIZE, cnt[0], bid);
        arrive_inc(signals);
        wait_eq_reset(signals);
        // d2d: from temp
        copy(temp, d_values, s_data, 0, d_buffer_size, d_buffer_offset, d_buffer_stride, cnt[0], MAP_SIZE, cnt[0], bid);
    }
}

#endif // COPY_CUH
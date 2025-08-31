"""
本文件（cache_utils.py）定义了 Transformers 中文本生成相关的「层级缓存（layer-level cache）」抽象与具体实现，用于在自回归解码中高效复用注意力的 Key/Value（KV）状态。

一、整体设计与术语
- 层级缓存（Layer Cache）：每一层自注意力都会维护一份 KV 缓存，用于在解码阶段避免重复计算历史 token 的注意力。
- 统一接口：通过抽象基类 CacheLayerMixin 统一定义缓存的关键操作：
  1) lazy_initialization(key_states): 延迟初始化缓存张量的形状、dtype、device 等；
  2) update(key_states, value_states, cache_kwargs): 追加或写入新的 KV，并返回用于注意力计算的 KV；
  3) get_mask_sizes(cache_position): 返回掩码计算所需的长度信息（kv_length 与 kv_offset）；
  4) get_seq_length(): 已缓存（或累计）序列长度；
  5) get_max_cache_shape(): 最大可容纳的序列长度（静态缓存/滑窗缓存用于上层构造注意力 mask）；
  6) 以及 offload/prefetch/reset/reorder_cache 等实用方法，用于设备迁移、清零复用、beam search 重排等典型流程。

- cache_position：上层会在每一步解码传入一个整数索引张量，指示“本次写入 KV 的位置（时间步）”。静态缓存会根据该位置原地写入；动态缓存则在末尾拼接（append）。
- 形状约定：key_states/value_states 一般为 [batch_size, num_heads, t, head_dim]，其中 t 是本次写入/参与注意力的新 token 数（解码阶段通常为 1，prefill 阶段可能为多个）。

二、三类核心缓存实现
1) 动态缓存 DynamicLayer（默认）
   - 通过在序列维度上拼接方式增长（cat），不设最大长度上限（get_max_cache_shape 返回 -1）。
   - 优点：使用简单，适配任何长度；缺点：频繁拼接带来额外开销，对 torch.compile 的“静态地址/图稳定”不友好。
   - get_mask_sizes：kv_length = past_seen_tokens + query_length，kv_offset = 0。
   - 还提供 crop/batch_repeat_interleave/batch_select_indices 等便捷操作，方便截断、复制与选择样本。

2) 静态缓存 StaticLayer（为 torch.compile 友好而设计）
   - 预先按照最大长度 max_cache_len 分配定长张量，随后使用 index_copy_ 将新 KV 原地写入对应槽位。
   - lazy_initialization 会在首次 update 时根据实际张量形状推断 batch、head、dtype、device 并分配背板存储；
   - 为保证编译图稳定，非 tracing 场景下会调用 torch._dynamo.mark_static_address 标记固定地址；
   - update 依赖 cache_kwargs["cache_position"] 写入对应位置；若缺省，则回退顺序写入 [0..t-1]；
   - get_mask_sizes：kv_length 固定返回 max_cache_len，kv_offset = 0；
   - get_seq_length：通过检测第一批第一个头在序列维度上的“非零占位”数量来近似推断已用长度。

3) 滑动窗口 SlidingWindowLayer（静态缓存的滑窗变体）
   - 仅保留最近窗口大小（sliding_window）的 KV；max_cache_len 取 min(sliding_window, 预设上限)。
   - prefill（提示词较长）时：若本次写入超过窗口长度，会缓存“最后 max_cache_len 段”，但返回值仍是完整 KV（便于继续注意力计算）。
   - 解码（通常 1 token/step）时：当缓存已满，整体左移一位（roll），再将新 token 写入最后一格，实现滑窗；
   - get_mask_sizes：基于 cache_position 计算 kv_offset（窗口外历史视作偏移），kv_length 取 max(query_length, max_cache_len)。

补充：
- DynamicSlidingWindowLayer 是“动态缓存 + 滑窗”的组合；
- ChunkedSlidingLayer 在滑窗基础上增加了 prefill 分块（chunking）优化，适配如 Llama 4 的大规模前填充策略，降低峰值显存并提升吞吐。

三、与上层生成流程的协作
- 生成入口（generate/greedy/beam 等）通常会创建或传递 past_key_values；
- 静态缓存需要上层提供 cache_position 来指示“将新 token 的 KV 放到第几个时间步”；
- get_mask_sizes 的返回（kv_length、kv_offset）被上层用于构造注意力 mask，确保仅对合法历史进行注意力；
- reorder_cache / batch_* 等在 beam search、采样扩展与剪枝时被调用，保证缓存与批次对齐。

四、何时选用哪种缓存
- 追求易用与最大灵活性（不限长度/不固定图）：DynamicLayer；
- 追求编译优化（torch.compile）、稳定地址与高吞吐：StaticLayer / SlidingWindowLayer；
- 需要长上下文但仅关注最近窗口：SlidingWindow 或其动态/分块变体；
- 注意：静态缓存需要在初始化时给出 max_cache_len（通常与 config.max_position_embeddings 或预期最大解码步数一致）。

五、性能与内存权衡
- 动态缓存：内存随实际长度增长，但存在拼接与 Python 调度开销；
- 静态缓存：一次性分配固定显存，更新为原地写入，利于编译器优化与 CUDA Graph，整体吞吐更优；
- 滑窗：将历史裁剪为固定窗口，节省显存与注意力复杂度，适合长上下文流式推理。

六、跨设备/复用与兼容性
- offload/prefetch 用于在 CPU/GPU 间迁移缓存；reset 可在复用同一缓存对象时快速清零；
- 某些设备（如 MPS）不支持 index_copy_，实现中提供了切片赋值回退；
- 兼容旧模型：在 cross-attention 场景可能不传 cache_kwargs 或传 None，StaticLayer 会回退到顺序写入 [0..t-1]。

阅读指引：
- CacheLayerMixin 定义抽象接口；
- DynamicLayer / DynamicSlidingWindowLayer：按需增长的实现；
- StaticLayer / SlidingWindowLayer / ChunkedSlidingLayer：预分配+原地写入，适配 compile/graph；
- Cache 及其子类（DynamicCache/StaticCache/SlidingWindowCache/HybridCache/QuantizedCache/EncoderDecoderCache 等）
  负责“跨层”管理与封装，提供更新、重排、裁剪、批次操作、早期初始化等高层接口。
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from .configuration_utils import PretrainedConfig
from .utils import (
    is_hqq_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


if _is_quanto_greater_than_0_2_5 := is_quanto_greater("0.2.5", accept_dev=True):
    from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys, self.values = None, None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.keys is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.keys is not None and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=True)
            self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        # Compute the full states
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]

        kv_offset = torch.clamp(first_cache_position - self.sliding_window + 1, min=0)

        if self.get_seq_length() >= self.sliding_window:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.get_seq_length() + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
        """
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.keys is not None else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class SlidingWindowLayer(StaticLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing
    tensors, and then mutates them in-place. Built for `torch.compile` support.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        """
        Args:
            max_cache_len (`int`):
                Maximum number of tokens that can be stored, used for tensor preallocation.
            sliding_window (`int`):
                The size of the sliding window.
        """
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position")

        is_full = self.cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        # Handle prefill phase when prompt length > sliding_window_size.
        # Note that we store cropped key/value states in the cache but return the full key/value states.
        if cache_position.shape[0] > self.max_cache_len:
            self.keys.copy_(key_states[:, :, -self.max_cache_len :, :])
            self.values.copy_(value_states[:, :, -self.max_cache_len :, :])
            # Return the full states here
            return key_states, value_states

        # Here we only assume decoding stage, i.e. 1 token at a time
        if is_full:
            # Roll all values to the left by 1 position
            new_keys = self.keys.roll(-1, dims=-2)
            new_values = self.values.roll(-1, dims=-2)
            # Overwrite the last position with new states
            # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
            index = torch.tensor([-1], dtype=int, device=self.device)
            new_keys[:, :, index] = key_states
            new_values[:, :, index] = value_states

            # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
            self.keys.copy_(new_keys)
            self.values.copy_(new_values)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]

        kv_offset = torch.clamp(first_cache_position - self.max_cache_len + 1, min=0)
        # This is not general (see HybridChunkedCache for the whole general case), but it's what the cache returns
        kv_length = max(query_length, self.max_cache_len)
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class ChunkedSlidingLayer(SlidingWindowLayer):
    """
    An extended SlidingWindowLayer that supports prefill chunking, originally implemented for Llama 4.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position")

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += key_states.shape[-2]

        if is_full:
            full_key_states = torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
            full_value_states = torch.cat((self.values[:, :, 1:, :], value_states), dim=-2)
            # Fast decoding path -> here as the effective size is still sliding window, it is extremely important
            # to return `self.key_cache[layer_idx]` and `self.value_cache[layer_idx]`, as they have the fixed address
            # in memory (the values are the same as the full states, but not the address!!)
            if key_states.shape[-2] == 1:
                self.keys.copy_(full_key_states)
                self.values.copy_(full_value_states)
                return self.keys, self.values
        elif not is_full and cumulative_length + key_states.shape[2] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((self.keys[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states
            return self.keys, self.values

        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.keys/values` here, as otherwise we lose some context
        # which is outside the window
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        first_cache_position = cache_position[0]
        sliding_window = self.max_cache_len

        kv_offset = torch.clamp(first_cache_position - sliding_window + 1, min=0)
        # This is the true general case for any Cache using local attention (sliding or chunked)
        if first_cache_position >= sliding_window:
            # Here the Cache is already full
            kv_length = sliding_window + query_length - 1
        elif first_cache_position < sliding_window and first_cache_position + query_length > sliding_window:
            # Here the Cache becomes full with the new input
            kv_length = first_cache_position + query_length
        else:
            # Here the Cache is still smaller than the local size, but we return the local size as it's static
            kv_length = sliding_window
        return kv_length, kv_offset


class QuantizedLayer(DynamicLayer):
    """
    A quantized layer similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for the key and value caches by
    applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length`
    is set as a maximum capacity for the original precision cache. When the length goes beyond maximum capacity, the original
    precision cache is discarded and moved into the quantized cache. The quantization is done per-channel with a set `q_group_size`
    for both Keys and Values, in contrast to what was described in the paper.
    """

    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(self)
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necesary kes and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        self.cumulative_length += key_states.shape[-2]

        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)
            self._quantized_keys = self._quantize(key_states.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(value_states.contiguous(), axis=self.axis_value)
            return key_states, value_states

        dequant_keys = self._dequantize(self._quantized_keys)
        dequant_values = self._dequantize(self._quantized_values)
        keys_to_return = torch.cat([dequant_keys, self.keys, key_states], dim=-2)
        values_to_return = torch.cat([dequant_values, self.values, value_states], dim=-2)
        if self.keys.dim() == 4 and self.keys.shape[-2] + 1 >= self.residual_length:
            self._quantized_keys = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
            self._quantized_values = self._quantize(values_to_return.contiguous(), axis=self.axis_value)
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return keys_to_return, values_to_return

    @abstractmethod
    def _quantize(self, tensor, axis): ...

    @abstractmethod
    def _dequantize(self, q_tensor): ...

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class QuantoQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        if not _is_quanto_greater_than_0_2_5:
            raise ImportError(
                "You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCache`. "
                "Detected version {optimum_quanto_version}."
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        if not is_hqq_available():
            raise ImportError("You need to install `hqq` to use `HQQQuantizedLayer`")

        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.keys.device,
            compute_dtype=self.keys.dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.keys.dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.keys.device)  # Move to device and cast to dtype
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


STATIC_LAYER_CLASS_MAPPING: dict[str, type[CacheLayerMixin]] = {
    "full_attention": StaticLayer,
    "sliding_attention": SlidingWindowLayer,
    "chunked_attention": ChunkedSlidingLayer,
}


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Parameters:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[type[CacheLayerMixin]] = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate
        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the begining
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        # Prefetch
        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            self.layers[layer_idx].prefetch()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if not (only_non_sliding and self.is_sliding[layer_idx]):
            self.layers[layer_idx].offload()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as device and dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape, dtype and device), which is very efficient and practical
        fake_keys_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    DynamicCache()
    ```
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[PretrainedConfig] = None,
    ):
        """
        Create a `DynamicCache`. Specialized constructor for DDP cache data, needed for BC.

        Args:
            ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
                It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
                `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
                for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
                Note: it needs to be the 1st arg as well to work correctly
            config (`PretrainedConfig`, *optional*):
                The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
                or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
                `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        """
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            config = config.get_text_config()
            sliding_window = getattr(config, "sliding_window", None) or getattr(config, "attention_chunk_size", None)
            layer_types = getattr(config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(config.num_hidden_layers)
                ]

            for layer_type in layer_types:
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a DynamicLayer for each entry of the ddp_data
                if config is None:
                    layers.append(DynamicLayer())
                # Update the layer with the data
                _, _ = layers[layer_idx].update(key_states, value_states)

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(layer_class_to_replicate=DynamicLayer)
        else:
            super().__init__(layers=layers)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]]) -> "DynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


# Utilities for `DynamicCache` <> torch.export support

if is_torch_greater_or_equal("2.3"):

    def _get_cache_dict(cache: DynamicCache):
        if any(not isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)) for layer in cache.layers):
            raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

        if not is_torch_greater_or_equal_than_2_6:
            logger.warning_once(
                "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
            )

        return {
            "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
            "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
        }

    def _unflatten_dynamic_cache(
        values,
        context: torch.utils._pytree.Context,
    ):
        dictionary = torch.utils._pytree._dict_unflatten(values, context)
        cache = DynamicCache()
        # Reconstruct layers from keys and values lists
        key_list = dictionary.get("key_cache", [])
        value_list = dictionary.get("value_cache", [])
        for idx in range(max(len(key_list), len(value_list))):
            key = key_list[idx] if idx < len(key_list) else None
            value = value_list[idx] if idx < len(value_list) else None
            cache.update(key, value, idx)
        return cache

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
            _get_cache_dict(dynamic_cache)
        ),
    )
    # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache, lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec)
    )


class OffloadedCache(Cache):
    """
    A drop-in replacement for DynamicCache that conserves accelerator (GPU, XPU) memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(self) -> None:
        super().__init__(layer_class_to_replicate=DynamicLayer, offloading=True)


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in args and kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


class OffloadedStaticCache(Cache):
    """
    A drop-in replacement for StaticCache that conserves accelerator memory by offloading
    cache tensors to CPU when not actively being used.

    This cache maintains the compilation-friendly properties of StaticCache while enabling
    much longer sequences by offloading inactive layers to CPU memory.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

    >>> # Prepare a cache class with offloading
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = OffloadedStaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache with offloaded layers
    OffloadedStaticCache()
    ```
    """

    # Pass-in args and kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers, offloading=True)


class SlidingWindowCache(Cache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

    >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = SlidingWindowCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    SlidingWindowCache()
    ```
    """

    # Pass-in args and kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        layers = [SlidingWindowLayer(max_cache_len, config.sliding_window) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for models that alternate between a local sliding window
    attention and global attention in every other layer (originally implemented for Gemma2).
    Under the hood, Hybrid Cache leverages ["SlidingWindowLayer"] for sliding window attention and ["StaticLayer"]
    for global attention. For more information, see the documentation of those layer types.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

    >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = HybridCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    HybridCache()
    ```
    """

    # Pass-in args and kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        if hasattr(config, "layer_types"):
            layers = []
            for layer_type in config.layer_types:
                init_kwargs = {"max_cache_len": max_cache_len}
                if layer_type == "sliding_attention":
                    init_kwargs["sliding_window"] = config.sliding_window
                elif layer_type == "chunked_attention":
                    init_kwargs["sliding_window"] = config.attention_chunk_size
                layers.append(STATIC_LAYER_CLASS_MAPPING[layer_type](**init_kwargs))
        else:
            # In this case, fall back to StaticCache
            layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)


# The mapping already handles dispatching the correct layers in Hybrid, this is only used for BC
class HybridChunkedCache(HybridCache): ...


class OffloadedHybridCache(Cache):
    """
    A drop-in replacement for HybridChunkedCache that conserves accelerator memory by offloading
    cache tensors to CPU when not actively being used.

    This cache maintains the compilation-friendly properties of HybridChunkedCache while enabling
    much longer sequences by offloading inactive layers to CPU memory.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    # Pass-in args and kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(self, config: PretrainedConfig, max_cache_len: int, *args, **kwargs):
        if hasattr(config, "layer_types"):
            layers = []
            for layer_type in config.layer_types:
                init_kwargs = {"max_cache_len": max_cache_len}
                if layer_type == "sliding_attention":
                    init_kwargs["sliding_window"] = config.sliding_window
                elif layer_type == "chunked_attention":
                    init_kwargs["sliding_window"] = config.attention_chunk_size
                layers.append(STATIC_LAYER_CLASS_MAPPING[layer_type](**init_kwargs))
        else:
            # In this case, fall back to StaticCache
            layers = [StaticLayer(max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers, offloading=True)


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.
    """

    def __init__(
        self,
        backend: str,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        if backend == "quanto":
            layer_class = QuantoQuantizedLayer
        elif backend == "hqq":
            layer_class = HQQQuantizedLayer
        else:
            raise ValueError(f"Unknown quantization backend `{backend}`")

        layers = [
            layer_class(nbits, axis_key, axis_value, q_group_size, residual_length)
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)


class QuantoQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> # Run pip install quanto first if you don't have it yet
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = QuantoQuantizedCache(config=model.config, nbits=4)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    QuantoQuantizedCache()
    ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__("quanto", config, nbits, axis_key, axis_value, q_group_size, residual_length)


class HQQQuantizedCache(QuantizedCache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> # Run pip install hqq first if you don't have it yet
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = HQQQuantizedCache(config=model.config, nbits=4, axis_key=1, axis_value=1)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    HQQQuantizedCache()
    ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__("hqq", config, nbits, axis_key, axis_value, q_group_size, residual_length)


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Example:

    ```python
    >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
    >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

    >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

    >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
    >>> self_attention_cache = DynamicCache()
    >>> cross_attention_cache = DynamicCache()
    >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    EncoderDecoderCache()
    ```
    """

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.layers[layer_idx].keys,
                self.self_attention_cache.layers[layer_idx].values,
                self.cross_attention_cache.layers[layer_idx].keys,
                self.cross_attention_cache.layers[layer_idx].values,
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search.
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)

    @property
    def is_sliding(self):
        return self.self_attention_cache.is_sliding

    @property
    def is_compileable(self) -> bool:
        return self.self_attention_cache.is_compileable


# ShadowKV adapters: integrate existing ShadowKV engines with the Cache API
# These adapters allow using ShadowKV within the same interface as other cache types in this module.
try:
    from .kv_cache import ShadowKVCache as _ShadowKVGPUImpl, ShadowKVCache_CPU as _ShadowKVCPUImpl
except Exception:
    _ShadowKVGPUImpl, _ShadowKVCPUImpl = None, None


class ShadowKVLayer(CacheLayerMixin):
    """
    Adapter for a single transformer layer that delegates storage and retrieval to a shared ShadowKV engine.

    Required cache_kwargs for update():
    - phase: Literal["prefill", "decode"] (optional; inferred by incoming length if not provided)
    - For prefill: key_states_roped (torch.Tensor), query_states (optional)
    - For decode: query_states (torch.Tensor), rope_func (callable), cos_sin_cache (torch.Tensor)
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, engine, layer_idx: int):
        super().__init__()
        self.engine = engine
        self.layer_idx = layer_idx
        self.device = None
        self.dtype = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        # No local full KV storage; ShadowKV manages persistent storage.
        self.keys, self.values = None, None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.engine is None:
            raise RuntimeError("ShadowKV engine is not available. Make sure kernels are built and imports succeed.")

        if self.device is None:
            self.lazy_initialization(key_states)

        cache_kwargs = cache_kwargs or {}
        incoming = key_states.shape[-2]
        phase = cache_kwargs.get("phase") or ("prefill" if incoming > 1 else "decode")

        if phase == "prefill":
            key_states_roped = cache_kwargs.get("key_states_roped")
            # last_query_states is optional
            last_query_states = cache_kwargs.get("query_states")
            if key_states_roped is None:
                # Fallback: use provided keys directly if roped version isn't supplied
                key_states_roped = key_states
            # Store value on CPU side and compute landmarks; return original KV for attention in prefill
            if hasattr(self.engine, "prefill_kv_cache"):
                self.engine.prefill_kv_cache(
                    new_v_cache=value_states,
                    layer_idx=self.layer_idx,
                    key_states_roped=key_states_roped,
                    last_query_states=last_query_states,
                )
            # Return the full prefill KV as-is (full attention in prefill)
            self.keys, self.values = key_states, value_states
            return self.keys, self.values

        # decode phase: build sparse retrieval + append the new token into generation region
        query_states = cache_kwargs.get("query_states")
        if query_states is None:
            raise ValueError("ShadowKV decode requires 'query_states' in cache_kwargs.")
        rope_func = cache_kwargs.get("rope_func")
        cos_sin_cache = cache_kwargs.get("cos_sin_cache")

        # 1) retrieval indices based on current query
        position_ids = self.engine.get_retrieval_position_ids(layer_idx=self.layer_idx, query_states=query_states)
        # 2) gather value cache (CPU->GPU) into sparse region
        values = self.engine.get_value_cache(self.layer_idx, position_ids)
        # 3) reconstruct keys (SVD) + apply RoPE
        keys = self.engine.get_key_cache(self.layer_idx, position_ids, rope_func, cos_sin_cache)
        # 4) append new kv into generation region
        self.engine.update_kv_cache(new_k_cache=key_states, new_v_cache=value_states, layer_idx=self.layer_idx)

        self.keys, self.values = keys, values
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        # ShadowKV returns a composed KV for attention; we set offset=0 and length as past decoded + current query
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_decoded = self.get_seq_length()
        kv_length = past_decoded + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.engine is None:
            return 0
        # Global across layers in ShadowKV; sufficient for mask sizing in decode
        return int(self.engine.get_kv_len()) if hasattr(self.engine, "get_kv_len") else 0

    def get_max_cache_shape(self) -> int:
        # No fixed maximum from this layer's perspective
        return -1

    # Optional properties for integration with Cache.max_batch_size and Cache.max_cache_len
    @property
    def max_batch_size(self) -> int:
        return int(getattr(self.engine, "batch_size", -1)) if self.engine is not None else -1

    @property
    def max_cache_len(self) -> int:
        return int(getattr(self.engine, "max_length", -1)) if self.engine is not None else -1

    # No-op implementations to satisfy Cache's optional layer-level operations
    def crop(self, max_length: int) -> None:
        # Delegate cropping to the underlying ShadowKV engine if available.
        if self.engine is not None and hasattr(self.engine, "crop"):
            self.engine.crop(max_length)
        return None

    def batch_repeat_interleave(self, repeats: int) -> None:
        # No local storage duplication; batch layout is tracked by the engine.
        return None

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        # Selection is handled at higher levels or by the engine; nothing layer-local to update.
        return None

    def offload(self):
        # Offloading is managed by adapters/engine; nothing to do per-layer.
        return None

    def prefetch(self):
        # Prefetch is managed by adapters/engine; nothing to do per-layer.
        return None

    def reset(self) -> None:
        # Drop local references; engine clearing is handled by adapter.reset()
        self.keys, self.values = None, None
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        # Delegate beam reordering to the underlying ShadowKV engine if available.
        if self.engine is not None and hasattr(self.engine, "reorder_cache"):
            self.engine.reorder_cache(beam_idx)
        return None


class ShadowKVCacheAdapter(Cache):
    """
    Cache manager that wires per-layer ShadowKVLayer to a shared ShadowKV GPU engine.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        sparse_budget: int,
        chunk_size: int,
        rank: int,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
    ) -> None:
        if _ShadowKVGPUImpl is None:
            raise RuntimeError("ShadowKV GPU implementation is unavailable. Please build/install kernels.")
        engine = _ShadowKVGPUImpl(
            config=config,
            batch_size=batch_size,
            max_length=max_cache_len,
            device=device,
            dtype=dtype,
            sparse_budget=sparse_budget,
            chunk_size=chunk_size,
            rank=rank,
        )
        layers = [ShadowKVLayer(engine, i) for i in range(config.num_hidden_layers)]
        self.engine = engine
        super().__init__(layers=layers)

    def reset(self):
        if hasattr(self, "engine") and self.engine is not None and hasattr(self.engine, "clear"):
            self.engine.clear()
        super().reset()


# Provide a unified class name following other Cache variants
class ShadowKVCache(ShadowKVCacheAdapter):
    pass


class ShadowKVCacheCPUAdapter(Cache):
    """
    Cache manager that wires per-layer ShadowKVLayer to a shared ShadowKV CPU-optimized engine.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        sparse_budget: int,
        chunk_size: int,
        rank: int,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
        local_chunk: int = 1,
        outlier_chunk: int = 0,
    ) -> None:
        if _ShadowKVCPUImpl is None:
            raise RuntimeError("ShadowKV CPU implementation is unavailable. Please build/install kernels.")
        engine = _ShadowKVCPUImpl(
            config=config,
            batch_size=batch_size,
            max_length=max_cache_len,
            device=device,
            dtype=dtype,
            sparse_budget=sparse_budget,
            chunk_size=chunk_size,
            rank=rank,
        )
        layers = [ShadowKVLayer(engine, i) for i in range(config.num_hidden_layers)]
        self.engine = engine
        super().__init__(layers=layers)

    def reset(self):
        if hasattr(self, "engine") and self.engine is not None and hasattr(self.engine, "clear"):
            self.engine.clear()
        super().reset()


### Deprecated classes


class SinkCache(Cache):
    """
    It is now a `custom_generate` repository on the Hub: https://huggingface.co/transformers-community/sink_cache.
    See [these docs](https://huggingface.co/docs/transformers/generation_strategies#custom-decoding-methods) for
    general `custom_generate`usage.
    """

    # TODO (joao, manuel): Remove this class in v4.59.0
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "`SinkCache` has been moved as a `custom_generate` repository on the Hub: "
            "https://huggingface.co/transformers-community/sink_cache. See the repository for usage examples."
        )

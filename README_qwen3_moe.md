# Qwen3MoE ShadowKV Implementation

## 概述

本项目实现了 Qwen3MoE 模型的 ShadowKV 优化版本，基于原有的 Qwen2 实现进行扩展，支持混合专家系统（MoE）架构和稀疏注意力机制。

## 文件说明

### `models/qwen3_moe.py`

主要实现文件，包含以下核心组件：

#### `Qwen3MoeLayer` 类
- 存储单层 Transformer 的权重参数
- 支持 MoE 层和标准 MLP 层
- 包含 Q/K 归一化机制（Qwen3MoE 特有）
- 支持独立的 QKV 投影权重和偏置项

#### `Qwen3Moe` 类
- 继承自 `LLM` 基类
- 实现 Qwen3MoE 模型的高效推理
- 支持多种优化策略：
  - 标准全注意力模式
  - ShadowKV 稀疏注意力模式
  - 高效的 KV 缓存管理
  - MoE 专家系统优化

## 主要特性

### 1. ShadowKV 稀疏注意力
- 通过 SVD 分解识别重要的键值对
- 显著降低内存占用
- 支持超长上下文处理（最大 64K tokens）

### 2. MoE 架构支持
- 混合专家系统实现
- 动态专家路由
- 负载均衡优化

### 3. Q/K 归一化
- Qwen3MoE 特有的 Q/K 归一化机制
- 提高训练稳定性和推理质量

### 4. 高效内存管理
- 支持 CPU-GPU 混合存储
- 动态 KV 缓存管理
- 内存优化的批量推理

### 5. RoPE 位置编码
- 旋转位置编码支持
- 预计算缓存优化
- 支持超长序列

## 核心方法

### Qwen3MoeLayer 类
- `__init__(layer_idx)`: 初始化层索引
- `init_parameters(hf_layer)`: 从HuggingFace层加载权重参数
- `init_gpu(device)`: 将权重转移到GPU设备

### Qwen3Moe 类
- `__init__(...)`: 初始化模型配置和参数
- `init_parameters()`: 加载预训练模型权重
- `pre_attention_compute(...)`: 注意力前预处理（QKV投影、Q/K归一化）
- `post_attention_compute(...)`: 注意力后处理（MoE前馈网络）
- `_moe_forward(...)`: MoE专家路由和计算
- `apply_rotary_pos_emb(...)`: 应用RoPE位置编码
- `apply_rotary_pos_emb_single(...)`: 单张量RoPE编码
- `generate(...)`: 文本生成（继承自LLM基类）
- `chat(...)`: 对话生成（继承自LLM基类）
- `inference(...)`: 模型推理（继承自LLM基类）

## 使用示例

```python
from models.qwen3_moe import Qwen3Moe

# 初始化模型
model = Qwen3Moe(
    model_name="Qwen/Qwen2.5-MoE-A2.7B-Instruct",
    batch_size=1,
    max_length=32768,
    device="cuda",
    dtype=torch.float16,
    attn_mode="shadowkv",
    sparse_budget=4096
)

# 文本生成
response = model.generate(
    input_ids=input_tokens,
    gen_len=256,
    temperature=0.7
)

# 对话
response = model.chat(
    "你好，请介绍一下 ShadowKV 技术。",
    gen_len=512
)
```

## 技术优势

1. **内存效率**: ShadowKV 技术显著降低长上下文推理的内存占用
2. **推理速度**: 优化的注意力计算和 KV 缓存管理
3. **模型兼容**: 完全兼容 HuggingFace Qwen3MoE 模型
4. **扩展性**: 支持批量推理和超长上下文
5. **稳定性**: Q/K 归一化提高数值稳定性

## 依赖要求

- PyTorch >= 1.13
- transformers >= 4.21
- flash-attn >= 2.0
- 其他依赖见 `requirements.txt`

## 测试验证

运行测试脚本验证实现：

```bash
python test_qwen3_moe.py
```

测试内容包括：
- 语法编译检查
- 类定义验证
- 方法完整性检查
- MoE 特性验证

## 注意事项

1. 首次运行需要下载 HuggingFace 模型权重
2. ShadowKV 模式需要足够的 GPU 内存进行 SVD 计算
3. MoE 模型的专家数量会影响内存占用
4. 建议使用 FP16 或 BF16 精度以节省内存

## 贡献

本实现基于 ShadowKV 论文和 Qwen2 的实现，针对 Qwen3MoE 架构进行了适配和优化。
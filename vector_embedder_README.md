# BGE Large 中文向量嵌入器（本地加载 / CPU & GPU 通用）

本模块提供基于 **BAAI/bge-large-zh-v1.5** 的文本向量化能力，**强制从本地 Hugging Face 缓存快照加载模型**，并同时支持 CPU / GPU 推理。

特性：

* ✅ CPU 推理（默认）
* ✅ GPU 推理（可选，`use_gpu=True` 且 CUDA 可用时生效）
* ✅ 强制本地离线加载（不联网）
* ✅ 批量向量化
* ✅ 线程安全（模型加载锁 + 推理锁）
* ✅ L2 归一化输出，适合向量检索/相似度计算

---

## 1. 环境依赖

Python `>= 3.8`

安装依赖：

```bash
pip install torch transformers
```

> 若你有 GPU 并希望使用 CUDA 版 torch，请按 PyTorch 官网对应命令安装；否则默认 CPU 版即可。

---

## 2. 本地模型目录说明

Hugging Face 模型缓存路径类似：

```
C:\Users\<用户名>\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5\
```

你需要传入 **snapshots 下的完整快照目录**（包含 config / tokenizer / 权重文件），例如：

```
...snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116
```

该快照目录内应至少包含：

* `config.json`
* `pytorch_model.bin` 或 `model.safetensors`
* `tokenizer.json / vocab.txt / tokenizer_config.json / special_tokens_map.json`

> 注意：snapshots 下面可能有多个 hash。请选**文件最完整（config + tokenizer + 权重齐全）**的那一个。

---

## 3. 快速开始

### 3.1 CPU 模式（默认）

```python
from utils.vector_embedder import BgeLargeEmbedder

local_path = r"C:\Users\<用户>\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116"

embedder = BgeLargeEmbedder(
    local_model_path=local_path,
    use_gpu=False,   # CPU 模式
    dtype=None,      # CPU 推荐 None（float32）
    pooling="mean"
)
```

### 3.2 GPU 模式（如果你有 GPU）

```python
import torch
from utils.vector_embedder import BgeLargeEmbedder

local_path = r"C:\Users\<用户>\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116"

embedder = BgeLargeEmbedder(
    local_model_path=local_path,
    use_gpu=True,            # GPU 模式
    dtype=torch.float16,     # GPU 推荐 float16（省显存 + 提速）
    pooling="mean"
)
```

> 如果 `use_gpu=True` 但检测不到 CUDA，会自动回退到 CPU。

---

## 4. 使用示例

### 4.1 单条文本 embedding

```python
vec = embedder.embed_query("你好，世界")
print(len(vec), vec[:5])
```

### 4.2 批量 embedding

```python
texts = ["今天天气不错", "我想去北京旅游", "向量检索是什么"]
vecs = embedder.embed_texts(texts)
print(len(vecs), len(vecs[0]))
```

### 4.3 最小可运行示例（建议自测）

```python
if __name__ == "__main__":
    local_path = r"C:\Users\<用户>\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116"

    embedder = BgeLargeEmbedder(
        local_model_path=local_path,
        use_gpu=False,
        pooling="mean",
        lazy_load=False
    )

    vec = embedder.embed_query("测试一下本地 BGE 向量化")
    print("dim =", len(vec))
```

---

## 5. 参数说明

### `BgeLargeEmbedder(...)`

| 参数名                 | 类型            |                        默认值 | 说明                                                 |
| ------------------- | ------------- | -------------------------: | -------------------------------------------------- |
| `local_model_path`  | `str`         |                     **必填** | 本地快照目录（snapshots/<hash>），强制本地加载                    |
| `model_name`        | `str`         | `"BAAI/bge-large-zh-v1.5"` | 仅日志显示，不参与下载                                        |
| `use_gpu`           | `bool`        |                    `False` | 是否启用 GPU；True 时若 CUDA 不可用会自动用 CPU                  |
| `max_batch_size`    | `int`         |                       `64` | 单次推理 batch；CPU 可适当调小避免卡顿                           |
| `max_length`        | `int`         |                      `512` | tokenizer 最大截断长度                                   |
| `pooling`           | `str`         |                   `"mean"` | 池化方式：`"mean"`（推荐）或 `"cls"`                         |
| `dtype`             | `torch.dtype` |                     `None` | 推理精度：GPU 推荐 `torch.float16`；CPU 推荐 `None`（float32） |
| `trust_remote_code` | `bool`        |                    `False` | 是否信任远程代码（本地模式通常 False）                             |
| `lazy_load`         | `bool`        |                    `False` | 是否懒加载；True 时首次 embed 才加载模型                         |

---

## 6. 运行机制

### 6.1 强制本地加载

内部使用：

```python
AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
AutoModel.from_pretrained(local_model_path, local_files_only=True)
```

保证：

* 不联网
* 本地缺文件直接报错

### 6.2 CPU/GPU 自动选择

```python
self.device = torch.device(
    "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
)
```

* `use_gpu=False` → CPU
* `use_gpu=True` 且 CUDA 可用 → GPU
* `use_gpu=True` 但 CUDA 不可用 → 自动降级 CPU

### 6.3 embedding 生成流程

1. 拼接检索指令前缀：

   ```
   为这个句子生成表示以用于检索相关文本：<文本>
   ```

2. tokenizer 编码（padding + truncation）

3. 模型 forward 得到 token embeddings

4. pooling（mean/cls）

5. L2 normalize

6. 输出 python float list

---

## 7. 性能建议

### CPU 环境

* `max_batch_size=8~16`
* `max_length=256~512`
* 尽量批量推理

### GPU 环境

* `dtype=torch.float16`
* `max_batch_size` 可提高（按显存调）
* 自动混合精度 `autocast` 会启用提速

---

## 8. 常见问题

### Q1：snapshots 下为什么有多个 hash？

每个 hash 代表一个缓存快照版本。
**请选择包含 tokenizer + config + 权重完整文件的快照目录**作为 `local_model_path`。

### Q2：怎么确认不会联网？

断网运行即可验证。
本地缺文件会报错，但不会触发下载。
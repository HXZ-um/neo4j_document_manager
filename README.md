# Neo4j Document Manager

## 项目概述

Neo4j Document Manager 是一个基于 Neo4j 图数据库的智能文档管理工具，专为 Dify 平台设计。它提供了一套完整的文档处理和语义检索功能，支持将文档内容进行分块、向量化，并在 Neo4j 中进行高效存储和语义搜索。

### 核心特性

* **文档存储**：支持文本/PDF 分块向量化并存入 Neo4j
* **双索引语义查询**：同时检索 `chunk_embedding_index` 和 `properties_embedding_index`，支持加权合并
* **节点属性向量化**：任意标签节点（如 Equipment、Project）均可序列化后生成向量
* **插件化设计**：兼容 Dify 平台，提供 `neo4j_doc_store`、`neo4j_semantic_query` 等工具
* **线程安全与事务保证**：确保多线程下模型安全加载及数据一致性
* **本地离线模型**：BGE 模型强制从本地 Hugging Face 缓存加载，不依赖联网下载

---

## 技术架构

### 技术栈

* **Python 3.12+**：主开发语言
* **Neo4j 5.28.2**：图数据库，支持向量索引
* **BAAI/bge-large-zh-v1.5**：中文向量模型，用于语义嵌入
* **Transformers**：HuggingFace 库加载 BGE 模型
* **PyTorch 2.7.1**：深度学习框架，支持 CPU / GPU 推理
* **Dify Plugin Framework 0.2.x**：插件化架构支持
* **FastAPI**：高性能 Web 框架，提供 RESTful API 接口

### 项目结构

```
├── _assets/                # 配置文件（config.py）
├── provider/              # Dify插件入口
├── tools/                 # 工具实现
│   ├── neo4j_doc_store.py         # 文档存储工具
│   ├── neo4j_semantic_query.py    # 语义查询工具
│   └── neo4j_properties_embedded.py # 节点属性向量化工具
├── utils/                 # 核心模块
│   ├── vector_embedder.py         # BGE向量生成（强制本地加载）
│   ├── neo4j_store.py            # Neo4j操作封装
│   └── document_processor.py      # 文档分块处理
├── main.py               # FastAPI主程序入口
├── manifest.yaml         # Dify插件清单
└── requirements.txt      # 项目依赖
```

---

## 快速开始

### 环境要求

* **Python版本**：3.12+
* **内存**：至少 4GB（BGE-large 模型需求）
* **Neo4j版本**：5.x（支持向量索引）
* **GPU支持**：可选（若启用需 CUDA 环境）
* **模型缓存**：需提前在本地 Hugging Face 缓存中准备好 BGE 模型快照

### 安装依赖

```bash
pip install -r requirements.txt
```

---

## 配置说明

### 1) Neo4j 连接配置（Dify 插件运行时传入）

插件支持通过 Dify 平台界面配置 Neo4j 数据库连接参数：

1. 在 Dify 平台中安装插件后，进入插件设置页面
2. 填写以下配置项：

   * **Neo4j URI**：Neo4j 数据库地址（例如：bolt://localhost:7687）
   * **Neo4j Username**：连接数据库的用户名
   * **Neo4j Password**：连接数据库的密码

这些配置将在运行时传递给所有工具，无需在代码中硬编码。

### 2) BGE 本地模型路径配置（必须）

本项目的 BGE 向量模型采用 **强制本地加载**（不联网）。
你需要在 `_assets/config.py` 中配置本地模型快照路径：

```python
BGE_LARGE_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# 指向 snapshots/<hash> 的完整快照目录
BGE_LARGE_LOCAL_PATH = r"C:\Users\<用户名>\.cache\huggingface\hub\models--BAAI--bge-large-zh-v1.5\snapshots\<hash>"
```

> snapshots 下可能有多个 hash，请选择**包含 config + tokenizer + 权重文件完整的一套**。

---

## FastAPI Web服务使用说明

本项目除了作为 Dify 插件使用外，还提供一个基于 FastAPI 的独立 Web 服务，可以直接通过 HTTP API 调用所有功能。

### 启动服务

```bash
python main.py
```

服务默认运行在 `http://127.0.0.1:10086`，可在 `main.py` 的 `uvicorn.run()` 中修改主机与端口。

> 服务启动时只初始化 embedder；模型在首次向量化时加载（`lazy_load=True`）。
> 如果你希望启动即加载模型，可将 `lazy_load=False`。

### API文档

* **Swagger UI**: [http://127.0.0.1:10086/docs](http://127.0.0.1:10086/docs)
* **ReDoc**: [http://127.0.0.1:10086/redoc](http://127.0.0.1:10086/redoc)

### API端点

#### 1. 健康检查

* **URL**: `GET /health`
* **描述**: 检查服务运行状态
* **响应**:

  ```json
  {
    "status": "healthy",
    "message": "服务运行正常"
  }
  ```

#### 2. 配置数据库连接

* **URL**: `POST /configure`
* **描述**: 配置 Neo4j 数据库连接信息
* **请求参数**:

  ```json
  {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password"
  }
  ```
* **响应**:

  ```json
  {
    "status": "success",
    "message": "数据库连接配置成功"
  }
  ```

#### 3. 存储文档

* **URL**: `POST /store_document`
* **描述**: 将文档内容进行分块并向量化，存储到 Neo4j
* **请求参数**:

  ```json
  {
    "source_url": "https://example.com/document.pdf",
    "file_name": "document.pdf",
    "text": "文档内容文本"
  }
  ```
* **响应**:

  ```json
  {
    "status": "success",
    "message": "文件 document.pdf 处理成功，共插入 5 个文本块"
  }
  ```

#### 4. 语义查询

* **URL**: `POST /semantic_query`
* **描述**: 执行语义查询
* **请求参数**:

  ```json
  {
    "query_text": "查询内容",
    "search_type": "mixed",
    "top_k": 10,
    "chunk_weight": 0.7,
    "properties_weight": 0.3
  }
  ```
* **参数说明**:

  * `search_type`: 查询类型，可选值为 `"mixed"`(混合)、`"chunks"`(仅文本块)、`"nodes"`(仅节点)
  * `top_k`: 返回结果数量
  * `chunk_weight`: chunk 向量权重（仅 mixed 模式下生效）
  * `properties_weight`: properties 向量权重（仅 mixed 模式下生效）
* **响应**:

  ```json
  {
    "status": "success",
    "results": [...],
    "message": "混合查询成功，找到 8 条相关结果"
  }
  ```

#### 5. 节点属性向量化

* **URL**: `POST /node_properties`
* **描述**: 将节点属性序列化后生成向量嵌入并存储到 Neo4j
* **请求参数**:

  ```json
  {
    "label": "Equipment",
    "metadata_json": {"name": "设备名称", "type": "设备类型", "spec": "技术规格"}
  }
  ```
* **响应**:

  ```json
  {
    "status": "success",
    "message": "节点 Equipment 属性已成功向量化存储"
  }
  ```

---

## 工具说明

### 1. 文档存储工具 (neo4j_doc_store)

将文档内容分块并向量化，创建 Chunk 节点存储原文本内容以及向量嵌入，并关联到 Drawing 节点。

**参数：**

* `source_url` (string, 必填)：文档源 URL 或文件路径
* `file_name` (string, 必填)：文件名
* `text` (string, 必填)：要处理的文本内容

### 2. 语义查询工具 (neo4j_semantic_query)

通过文本查询向量索引，支持多种查询模式选择：

* **混合查询**：同时检索文本块和节点属性，支持加权合并
* **仅文本块查询**：只检索文档分块内容
* **仅节点查询**：只检索节点属性信息

**参数：**

* `search_type` (string, 可选, 默认"mixed")
* `query_text` (string, 必填)
* `top_k` (int, 可选, 默认10)
* `chunk_weight` (float, 可选, 默认0.7)
* `properties_weight` (float, 可选, 默认0.3)

### 3. 节点属性向量化工具 (neo4j_properties_embedded)

将实体节点属性序列化后向量化并存储到 Neo4j 中，支持任意标签节点。

**参数：**

* `label` (string, 必填)：节点标签
* `metadata_json` (string, 必填)：包含节点属性的 JSON 对象

---

## 技术细节

### 向量模型

项目使用 **BAAI/bge-large-zh-v1.5** 中文向量模型，特点如下：

* 专为中文语义检索优化
* 1024 维向量输出
* 支持 CPU / GPU 推理
* **本项目采用强制本地加载模式**：不会联网下载模型

  * 模型需提前缓存到本地 Hugging Face snapshots 中
  * 由 `BGE_LARGE_LOCAL_PATH` 指定快照路径

### Neo4j 数据模型

```
(:Drawing {fileName, uid, filePath}) 
  <-[:PART_OF]- (:Chunk {id, text, embedding, chunk_num})
```

* **Drawing 节点**：代表文档，包含文件名、唯一 ID 和路径
* **Chunk 节点**：代表文档分块，包含文本内容、向量嵌入和块编号
* **PART_OF 关系**：连接 Chunk 和 Drawing 节点

### 向量索引

Neo4j 中创建了两个向量索引以支持双索引语义查询：

1. **chunk_embedding_index**：用于 Chunk 节点的文本内容向量检索
2. **{label}_properties_embedding_index**：用于任意标签节点的属性向量检索（动态创建）

---

## 注意事项

1. **模型必须提前存在本地缓存**

   * 本项目不会自动下载模型
   * 如果本地快照不完整会直接报错
2. **内存要求**：BGE-large 模型加载建议至少 4GB 内存
3. **数据库配置**：确保 Neo4j 数据库已正确配置并可访问
4. **索引创建**：插件会自动创建必要的约束和向量索引

---

## 故障排查

### 常见问题

1. **模型加载失败 / 找不到 tokenizer 文件**：

   * 检查 `BGE_LARGE_LOCAL_PATH` 是否指向 `snapshots/<hash>`
   * 确认该快照目录包含完整文件（config + tokenizer + 权重）
   * snapshots 下若有多个 hash，换成“最完整”的那套

2. **数据库连接失败**：

   * 检查 Neo4j 服务是否运行
   * 验证连接参数（URI、用户名、密码）
   * 确认防火墙设置允许连接

3. **向量索引创建失败**：

   * 确认 Neo4j 版本为 5.x
   * 检查数据库权限设置
   * 查看 Neo4j 日志获取详细错误信息

### 测试验证

运行测试前请确保已正确配置 Neo4j 连接参数与本地模型路径。


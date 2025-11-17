# Neo4j Document Manager

**Author:** hxz  
**Version:** 0.0.1  
**Type:** 文档管理工具

## 项目概述

这是一个基于 Neo4j 图数据库的智能文档存储与管理工具，专为语义查询和图谱化文档存储而设计。该项目采用先进的 BGE-large-zh-v1.5 中文向量模型，提供高精度的文档语义理解和相似度检索能力。

本项目作为 Dify 平台的插件，提供了完整的文档管理解决方案，包括文档存储、语义查询和节点属性向量化等功能。

## 核心特性

- **🚀 智能向量化**: 使用 BAAI/bge-large-zh-v1.5 模型，专门优化中文文档的向量嵌入
- **📊 双索引存储**: 基于 Neo4j 图数据库，支持文本块和节点属性的分离式向量存储
- **🔍 语义查询**: 基于双向量索引的智能语义搜索，支持加权合并和结果排序
- **⚡ 高性能索引**: 支持 `chunk_embedding_index` 和 `properties_embedding_index` 双向量索引
- **🔄 数据管理**: 智能的文档更新机制，自动处理旧数据清理
- **🛡️ 线程安全**: 多线程环境下的安全模型加载和推理
- **⚖️ 加权合并**: 可配置的权重参数，灵活调节文本块与节点属性的优先级
- **📄 多格式支持**: 支持纯文本和PDF文档处理
- **🧩 插件化架构**: 与Dify平台无缝集成，提供多种工具

## 技术架构

### 核心技术栈
- **Python 3.12+**: 主要开发语言
- **Neo4j 5.28.2**: 图数据库，支持向量索引
- **Transformers**: BGE模型加载，使用AutoModel和AutoTokenizer
- **PyTorch 2.7.1**: 深度学习框架，支持GPU加速
- **Dify Plugin Framework 0.2.x**: 插件化架构支持

### 架构设计
- **模块化设计**: 各功能模块解耦，便于维护扩展
- **插件化架构**: 支持Dify平台集成
- **配置驱动**: 通过YAML配置文件灵活控制行为

## 快速开始

### 1. 环境准备

**系统要求:**
- Python 3.12+
- 内存 4GB+ （BGE-large模型要求）
- Neo4j 5.x 数据库实例

**安装依赖:**
```bash
# 克隆项目
git clone <repository-url>
cd neo4j_document_manager

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置数据库

编辑 `_assets/config.py` 文件，设置 Neo4j 数据库连接信息：

```python
# 文本处理配置
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Neo4j数据库配置
NEO4J_URI = "bolt://localhost:7687"  # 修改为你的Neo4j地址
NEO4J_USER = "neo4j"                 # 修改为你的用户名
NEO4J_PASSWORD = "your_password"     # 修改为你的密码

# 嵌入模型配置
BGE_LARGE_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
```

### 3. 初始化数据库

首次运行时，系统会自动创建必要的约束和向量索引。

### 4. 测试运行

```bash
# 测试BGE模型加载
python -c "from utils.vector_embedder import BgeLargeEmbedder; embedder = BgeLargeEmbedder('BAAI/bge-large-zh-v1.5'); print('模型加载成功!')"

# 启动主程序
python main.py
```

## 项目结构

```
neo4j_document_manager/
├── _assets/                    # 配置文件目录
│   └── config.py               # 核心配置文件
├── provider/                   # Dify插件提供者
│   ├── neo4j_document_manager.py    # 文档管理插件主程序
│   └── neo4j_document_manager.yaml  # 插件配置文件
├── tools/                      # Dify工具目录
│   ├── neo4j_doc_store.py           # 文档存储工具
│   ├── neo4j_doc_store.yaml         # 文档存储工具配置
│   ├── neo4j_semantic_query.py      # 语义查询工具
│   ├── neo4j_semantic_query.yaml    # 语义查询工具配置
│   ├── neo4j_properties_embedded.py  # 节点属性向量化工具
│   └── neo4j_properties_embedded.yaml # 节点属性向量化工具配置
├── utils/                      # 核心工具模块
│   ├── document_processor.py        # 文档处理模块
│   ├── neo4j_store.py              # Neo4j存储模块
│   └── vector_embedder.py           # BGE向量嵌入模块
├── main.py                     # 主程序入口
├── requirements.txt             # Python依赖列表
├── manifest.yaml                # Dify插件清单
└── README.md                   # 项目说明文档
```

### 模块说明

| 模块 | 功能描述 | 核心特性 |
|------|---------|----------|
| **vector_embedder.py** | BGE向量嵌入器 | • BGE检索指令前缀<br>• L2归一化处理<br>• GPU加速支持<br>• 线程安全设计 |
| **neo4j_store.py** | Neo4j图数据库操作 | • 双向量索引支持<br>• 文档更新机制<br>• 事务保证<br>• 加权查询合并 |
| **document_processor.py** | 文档处理器 | • 文本分块处理<br>• 语义完整性保持<br>• 可配置分块策略 |
| **neo4j_doc_store.py** | 文档存储工具 | • 双模式处理（文本+PDF）<br>• 自动向量化<br>• 事务安全存储<br>• 错误容错机制 |
| **neo4j_semantic_query.py** | 语义查询工具 | • 双索引相似度搜索<br>• 权重可配置<br>• 结果格式化<br>• 错误处理 |
| **neo4j_properties_embedded.py** | 节点属性向量化工具 | • 节点序列化<br>• 属性向量嵌入<br>• 支持任意标签 |

## 工具详解

### 📄 文档存储工具 (neo4j_doc_store)

#### 核心功能
- **双模式处理**：支持纯文本和PDF文件两种输入模式
- **智能分块**：使用可配置的分块大小和重叠度
- **BGE向量化**：采用BGE-large-zh-v1.5模型生成高质量中文向量
- **Neo4j存储**：创建Chunk节点存储原文本和向量，关联Drawing节点
- **事务安全**：确保数据一致性，支持文件更新时的旧数据清理

#### 参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `source_url` | string | ✅ | 文档源路径或URL |
| `file_name` | string | ✅ | 文件名标识符 |
| `text` | string | ✅ | 直接文本内容 |

#### 使用示例

**处理纯文本内容：**
```python
from tools.neo4j_doc_store import Neo4jDocStoreTool

tool = Neo4jDocStoreTool()
result = list(tool._invoke({
    "source_url": "manual_input",
    "file_name": "技术文档.txt",
    "text": "这是一个关于人工智能的技术文档内容..."
}))

print(result[0].message)  # 查看处理结果
```

**处理PDF文件：**
```python
result = list(tool._invoke({
    "source_url": "/path/to/document.pdf",
    "file_name": "设备手册.pdf",
    "text": "PDF文件内容..."  # 通过PDF处理提取的文本内容
}))
```

**处理在线PDF：**
```python
result = list(tool._invoke({
    "source_url": "https://example.com/manual.pdf",
    "file_name": "在线手册.pdf",
    "text": "在线PDF文件内容..."  # 通过PDF处理提取的文本内容
}))
```

#### 返回格式

成功处理时返回：
```json
{
  "status": "success",
  "message": "文件 'xxx' 处理成功，共插入 X 个文本块"
}
```

错误处理时返回：
```json
{
  "status": "error",
  "message": "具体错误信息"
}
```

#### 数据库结构

存储后的Neo4j图结构：
```cypher
(:Drawing {fileName: "技术文档.txt", uid: "hash_value", filePath: "manual_input"})
  ← [:PART_OF] - (:Chunk {id: "技术文档.txt_chunk1", text: "...", embedding: [...], chunk_num: 1})
  ← [:PART_OF] - (:Chunk {id: "技术文档.txt_chunk2", text: "...", embedding: [...], chunk_num: 2})
  ← [:PART_OF] - (...)
```

### 🔍 语义查询工具 (neo4j_semantic_query)

#### 功能概述
本工具基于双向量索引（`chunk_embedding_index`和`properties_embedding_index`）的智能语义搜索工具，支持加权合并和结果排序，可配置不同类型的查询权重。

#### 参数说明

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `search_type` | string | ❌ | mixed | 查询类型：mixed(混合)、nodes(仅节点)、drawings(仅图纸) |
| `query` | string | ✅ | - | 自然语言查询语句 |
| `chunk_weight` | number | ❌ | 0.4 | 文本块权重(0.0-1.0) |
| `node_weight` | number | ❌ | 0.6 | 节点属性权重(0.0-1.0) |
| `limit` | number | ❌ | 10 | 返回结果数量限制 |

#### 使用示例

**混合查询（默认）：**
```python
from tools.neo4j_semantic_query import Neo4jSemanticQueryTool

tool = Neo4jSemanticQueryTool()
result = list(tool._invoke({
    "query": "离心泵的安装方法",
    "search_type": "mixed",
    "chunk_weight": 0.4,
    "node_weight": 0.6,
    "limit": 10
}))

print(result[0].message)  # 查看处理结果
print(result[0].results)  # 查看查询结果
```

**仅节点属性查询：**
```python
result = list(tool._invoke({
    "query": "离心泵设备信息",
    "search_type": "nodes",
    "limit": 5
}))
```

**仅文档内容查询：**
```python
result = list(tool._invoke({
    "query": "设备安装步骤",
    "search_type": "drawings",
    "limit": 5
}))
```

#### 返回格式

成功查询时返回：
```json
{
  "status": "success",
  "message": "双索引查询完成，Chunk权重: 0.4, Node权重: 0.6",
  "results": [
    {
      "uid": "eq_001",
      "text": "设备安装步骤...",  // 文本块内容（仅chunk结果有）
      "similarity": 0.8567,    // 原始相似度分数
      "weighted_similarity": 0.7123,  // 加权后相似度分数
      "source": "node",        // 结果来源："chunk" 或 "node"
      "label": "Equipment",    // 主要节点标签
      "properties": {          // 节点所有属性
        "name": "离心泵",
        "type": "泵类设备",
        "manufacturer": "格兰富",
        "description": "用于输送清水的离心泵设备"
      }
    }
  ]
}
```

#### 查询结果格式

双索引查询返回的结果格式如下：

```json
{
  "uid": "eq_001",                    // 节点唯一标识符
  "text": "设备安装步骤...",        // 文本块内容（仅chunk结果有）
  "similarity": 0.8567,              // 原始相似度分数
  "weighted_similarity": 0.7123,     // 加权后相似度分数
  "source": "node",                   // 结果来源："chunk" 或 "node"
  "label": "Equipment",               // 主要节点标签
  "properties": {                     // 节点所有属性
    "name": "离心泵",
    "type": "泵类设备",
    "manufacturer": "格兰富",
    "description": "用于输送清水的离心泵设备"
  }
}
```

### 🧠 节点属性向量化工具 (neo4j_properties_embedded)

#### 功能概述
本工具用于将实体节点属性存储到Neo4j图数据库中，同时自动生成向量嵌入，支持节点属性的高效相似性搜索和语义分析。

#### 核心功能
- **自动向量化**: 将节点属性序列化后自动生成向量嵌入
- **灵活标签支持**: 支持任意节点标签（Equipment、Project等）
- **智能存储**: 自动创建向量索引并存储节点属性向量
- **属性保留**: 完整保留原始节点属性，便于后续查询和分析

#### 参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `label` | string | ✅ | Neo4j节点标签（实体类型） |
| `metadata_json` | string | ✅ | 节点属性的JSON字符串 |

#### 使用示例

**存储设备节点属性：**
```python
from tools.neo4j_properties_embedded import Neo4jPropertiesEmbeddedTool
import json

tool = Neo4jPropertiesEmbeddedTool()

# 设备节点属性
equipment_props = {
    "uid": "eq_001",
    "name": "离心泵",
    "type": "泵类设备",
    "manufacturer": "格兰富",
    "description": "用于输送清水的离心泵设备",
    "model": "CRN 32-4"
}

result = list(tool._invoke({
    "label": "Equipment",
    "metadata_json": json.dumps(equipment_props)
}))

print(result[0].message)  # 查看处理结果
```

**存储项目节点属性：**
```python
# 项目节点属性
project_props = {
    "uid": "proj_001",
    "name": "水处理系统升级项目",
    "status": "进行中",
    "description": "工厂水处理系统的设备升级改造项目",
    "startDate": "2025-01-15",
    "endDate": "2025-12-31"
}

result = list(tool._invoke({
    "label": "Project",
    "metadata_json": json.dumps(project_props)
}))
```

#### 返回格式

成功处理时返回：
```json
{
  "status": "success",
  "message": "节点 Equipment 属性已成功向量化存储到 properties_embedding_index"
}
```

错误处理时返回：
```json
{
  "status": "error",
  "message": "具体错误信息"
}
```

#### 数据库结构

存储后的Neo4j图结构：
```cypher
(:Equipment {
  uid: "eq_001",
  name: "离心泵",
  type: "泵类设备",
  manufacturer: "格兰富",
  description: "用于输送清水的离心泵设备",
  model: "CRN 32-4",
  properties_embedding: [...]  // 向量化后的属性嵌入
})
```

## 技术详细信息

### BGE模型特性
- **模型**: BAAI/bge-large-zh-v1.5
- **维度**: 1024维向量
- **语言**: 中文优化
- **检索指令**: 使用BGE专用前缀 `"为这个句子生成表示以用于检索相关文章："`
- **归一化**: L2归一化处理，提升相似度计算精度

### Neo4j图数据库设计
- **节点类型**:
  - `Drawing`: 文档节点，存储文件元信息
  - `Chunk`: 文本块节点，存储向量嵌入和文本内容
  - `Equipment/Project/等`: 任意业务节点，支持属性向量化
- **关系类型**:
  - `PART_OF`: Chunk属于Drawing的关系
- **索引类型**:
  - `chunk_embedding_index`: 文本块向量索引，支持余弦相似度搜索
  - `properties_embedding_index`: 节点属性向量索引，支持任意节点类型
  - 唯一性约束: Chunk.id 和 Drawing.fileName

### Dify插件集成
- **插件架构**: 基于Dify Plugin Framework 0.2.x构建
- **工具支持**: 提供三个核心工具（文档存储、语义查询、属性向量化）
- **配置驱动**: 通过YAML配置文件管理工具参数
- **API兼容**: 支持Dify平台的工具调用接口

### 性能优化
- **批量处理**: 支持批量生成向量，默认批大小64
- **GPU加速**: 可选启用GPU加速向量生成
- **混合精度**: 支持float16混合精度训练，节省显存
- **线程安全**: 多线程环境下的安全模型加载
- **懒加载**: 支持模型懒加载，提升启动速度

## 注意事项

### 系统要求
1. **模型下载**: 首次运行时会自动下载 BAAI/bge-large-zh-v1.5 模型（约1.3GB）
2. **内存要求**: BGE-large 模型需要至少 4GB 内存
3. **Neo4j 版本**: 需要 Neo4j 5.x 版本以支持向量索引功能
4. **PDF处理依赖**: 需要安装 PyPDF2 库（已包含在requirements.txt中）
5. **GPU 支持**: 可选择使用 GPU 加速（需要 CUDA 支持）

### 数据管理
- **文件更新机制**: 如果文件存在，系统会删除旧数据后插入新数据
- **事务保证**: 所有数据库操作都在事务中执行，保证数据一致性
- **连接管理**: 使用后请及时关闭数据库连接
- **资源清理**: neo4j_doc_store工具会自动管理资源释放

### 性能调优
- **批大小**: 根据内存大小调整 `max_batch_size` 参数
- **GPU使用**: 在有GPU的环境中设置 `use_gpu=True`
- **分块策略**: 根据文档类型调整 `chunk_size` 和 `chunk_overlap`
- **数据库连接池**: 高并发场景下建议配置Neo4j连接池

### 故障排查

#### 常见问题
1. **模型下载失败**: 检查网络连接，或手动下载到本地
2. **PyPDF2依赖错误**: 运行 `pip install PyPDF2` 安装依赖
3. **内存不足**: 减小 `max_batch_size` 或使用float16精度
4. **Neo4j连接失败**: 检查数据库连接配置和网络可达性
5. **向量索引错误**: 确保Neo4j版本支持向量索引（≥ 5.0）
6. **PDF解析失败**: 检查PDF文件是否损坏或加密

#### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 工具错误处理
`neo4j_doc_store` 工具提供详细的错误信息：
- **参数错误**: 明确指出缺少的必需参数
- **文件处理错误**: 提供具体的失败原因
- **数据库错误**: 返回详细的Neo4j操作失败信息
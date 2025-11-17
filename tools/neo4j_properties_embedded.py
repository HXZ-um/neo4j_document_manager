"""
节点属性向量化工具
将实体节点属性存储到Neo4j图数据库中，同时自动生成向量嵌入，支持节点属性的高效相似性搜索和语义分析
"""

import json
from collections.abc import Generator
from typing import Any, Dict

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

# 导入之前实现的模块
from utils.document_processor import DocumentProcessor
from utils.vector_embedder import BgeLargeEmbedder
from utils.neo4j_store import Neo4jVectorStore
from _assets.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    BGE_LARGE_EMBEDDING_MODEL,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)


def serialize_node(label: str, props: Dict[str, Any]) -> str:
    """
    将节点的标签和属性序列化为字符串，便于生成语义向量
    
    Args:
        label: 节点标签
        props: 节点属性字典
        
    Returns:
        序列化后的字符串
    """
    parts = [f"label: {label}"]
    for k, v in props.items():
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


class Neo4jPropertiesEmbeddedTool(Tool):
    """
    Neo4j节点属性向量化工具
    将实体节点属性序列化后生成向量嵌入并存储到Neo4j数据库中，支持任意标签的节点
    """
    
    processor = DocumentProcessor(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    embedder = BgeLargeEmbedder(model_name=BGE_LARGE_EMBEDDING_MODEL)

    store = Neo4jVectorStore(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    store.create_constraints()

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        执行节点属性向量化并存储到Neo4j
        
        Args:
            tool_parameters: 工具参数
                - label: 节点标签（必填）
                - metadata_json: 节点属性的JSON字符串（必填）
                
        Yields:
            执行结果消息
        """
        try:
            label = tool_parameters.get("label", "")
            if not label:
                yield self.create_json_message({
                    "status": "error",
                    "message": "节点标签不能为空",
                })
                return

            metadata_json = tool_parameters.get("metadata_json", "")

            # 基础参数校验
            if not metadata_json:
                yield self.create_json_message({
                    "status": "error",
                    "message": "json字符串不能为空",
                })
                return

            try:
                props = json.loads(metadata_json)

                # 先序列化节点
                serialized = serialize_node(label, props)

                # 计算向量
                embedding = self.embedder.embed_query(serialized)

                # 插入/更新到 Neo4j，使用新的 insert_node 方法
                success = self.store.insert_node(label=label, props=props, embedding=embedding)

                if success:
                    yield self.create_json_message({
                        "status": "success",
                        "message": f"节点 {label} 属性已成功向量化存储到 properties_embedding_index"

                    })
                else:
                    yield self.create_json_message({
                        "status": "error",
                        "message": f"节点 {label} 向量化存储失败",
                    })

            except json.JSONDecodeError as e:
                yield self.create_json_message({
                    "status": "error",
                    "message": f"JSON 解析失败：{str(e)}",
                })
            except Exception as e:
                yield self.create_json_message({
                    "status": "error",
                    "message": f"处理节点失败：{str(e)}",
                })

        except Exception as e:
            yield self.create_json_message({
                "status": "error",
                "message": f"存储失败：{str(e)}",
            })
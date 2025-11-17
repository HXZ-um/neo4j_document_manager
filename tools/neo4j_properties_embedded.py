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
        neo4j_store = None
        embedder = None
        
        try:
            # 获取参数
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

            # 获取运行时配置
            runtime_config = self.runtime.credentials if hasattr(self.runtime, 'credentials') else {}
            neo4j_uri = runtime_config.get('neo4j_uri', '')
            neo4j_user = runtime_config.get('neo4j_user', '')
            neo4j_password = runtime_config.get('neo4j_password', '')
            
            # 检查数据库配置
            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                yield self.create_json_message({
                    "status": "error",
                    "message": "错误：缺少数据库配置，请在插件设置中配置Neo4j连接信息",
                })
                return

            try:
                props = json.loads(metadata_json)

                # 先序列化节点
                serialized = serialize_node(label, props)

                # 初始化向量化器
                embedder = BgeLargeEmbedder(model_name=BGE_LARGE_EMBEDDING_MODEL)
                
                # 计算向量
                embedding = embedder.embed_query(serialized)

                # 初始化Neo4j存储
                neo4j_store = Neo4jVectorStore(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                )
                
                # 确保约束和索引存在
                neo4j_store.create_constraints()

                # 插入/更新到 Neo4j，使用新的 insert_node 方法
                success = neo4j_store.insert_node(label=label, props=props, embedding=embedding)

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

        finally:
            if neo4j_store:
                neo4j_store.close()
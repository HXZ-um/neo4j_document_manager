"""
语义查询工具
通过文本查询向量索引，支持加权合并chunk和properties的查询结果
"""

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils.neo4j_store import Neo4jVectorStore
from utils.vector_embedder import BgeLargeEmbedder
from _assets.config import BGE_LARGE_EMBEDDING_MODEL


class Neo4jSemanticQueryTool(Tool):
    """
    Neo4j语义查询工具
    通过文本查询向量索引，支持加权合并chunk和properties的查询结果
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        执行语义查询
        
        Args:
            tool_parameters: 工具参数
                - search_type: 查询类型（mixed混合、chunks仅文本块、nodes仅节点）默认mixed
                - query_text: 查询文本（必填）
                - top_k: 返回结果数量（可选，默认10）
                - chunk_weight: chunk向量权重（可选，默认0.7）
                - properties_weight: properties向量权重（可选，默认0.3）
                
        Yields:
            查询结果消息
        """
        neo4j_store = None
        
        try:
            # 获取参数
            search_type = tool_parameters.get('search_type', 'mixed')
            query_text = tool_parameters.get('query_text', '')
            top_k = tool_parameters.get('top_k', 10)
            chunk_weight = tool_parameters.get('chunk_weight', 0.7)
            properties_weight = tool_parameters.get('properties_weight', 0.3)
            
            # 获取运行时配置
            runtime_config = self.runtime.credentials if hasattr(self.runtime, 'credentials') else {}
            neo4j_uri = runtime_config.get('neo4j_uri', '')
            neo4j_user = runtime_config.get('neo4j_user', '')
            neo4j_password = runtime_config.get('neo4j_password', '')

            if not query_text:
                yield self.create_json_message({
                    "status": "error",
                    "message": "错误：查询文本不能为空"
                })
                return
                
            # 检查数据库配置
            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                yield self.create_json_message({
                    "status": "error",
                    "message": "错误：缺少数据库配置，请在插件设置中配置Neo4j连接信息",
                })
                return

            # 初始化Neo4j存储
            neo4j_store = Neo4jVectorStore(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )

            # 初始化向量化器
            embedder = BgeLargeEmbedder(
                model_name=BGE_LARGE_EMBEDDING_MODEL,
                use_gpu=False,
                max_batch_size=32,
                lazy_load=True
            )

            # 生成查询向量
            query_embedding = embedder.embed_query(query_text)
            
            # 根据查询类型执行不同的查询
            if search_type == 'chunks':
                # 仅查询文本块
                results = neo4j_store.search_chunks(query_embedding, top_k)
                message = f"文本块查询成功，找到 {len(results)} 条相关结果"
            elif search_type == 'nodes':
                # 仅查询节点
                results = neo4j_store.search_nodes(query_embedding, top_k)
                message = f"节点查询成功，找到 {len(results)} 条相关结果"
            else:
                # 混合查询（默认）
                results = neo4j_store.similarity_search(
                    query_embedding=query_embedding,
                    limit=top_k,
                    chunk_weight=chunk_weight,
                    node_weight=properties_weight
                )
                message = f"混合查询成功，找到 {len(results)} 条相关结果"

            if results:
                yield self.create_json_message({
                    "status": "success",
                    "results": results,
                    "message": message
                })
            else:
                yield self.create_json_message({
                    "status": "warning",
                    "results": [],
                    "message": "查询完成，但未找到相关结果"
                })

        except Exception as e:
            yield self.create_json_message({
                "status": "error",
                "message": f"查询失败：{str(e)}"
            })

        finally:
            if neo4j_store:
                neo4j_store.close()
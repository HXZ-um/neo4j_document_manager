"""
语义查询工具
基于双向量索引的智能语义搜索，支持加权合并和结果排序
"""

from collections.abc import Generator
from typing import Any, Optional
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils.vector_embedder import (
    BgeLargeEmbedder
)
from utils.neo4j_store import Neo4jVectorStore
from _assets.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BGE_LARGE_EMBEDDING_MODEL


class Neo4jSemanticQueryTool(Tool):
    """
    Neo4j语义查询工具
    基于双向量索引（chunk_embedding_index和properties_embedding_index）的智能语义搜索工具，
    支持加权合并和结果排序，可配置不同类型的查询权重
    """
    
    _embedder: Any = None
    _store: Optional[Neo4jVectorStore] = None

    def _get_embedder(self) -> Any:
        """
        获取或初始化向量嵌入器实例
        
        Returns:
            BgeLargeEmbedder实例
        """
        if self._embedder is None:
            self._embedder = BgeLargeEmbedder(
                model_name=BGE_LARGE_EMBEDDING_MODEL,
                use_gpu=False,  # 可根据实际情况调整
                max_batch_size=32,  # 批量大小优化
                lazy_load=True  # 启用懒加载提升启动速度
            )
        return self._embedder

    def _get_store(self) -> Neo4jVectorStore:
        """
        获取或初始化Neo4j存储实例
        
        Returns:
            Neo4jVectorStore实例
        """
        if self._store is None:
            self._store = Neo4jVectorStore(uri=NEO4J_URI, user=NEO4J_USER,password=NEO4J_PASSWORD)
        return self._store

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        执行语义查询
        
        Args:
            tool_parameters: 工具参数
                - search_type: 查询类型（mixed混合、nodes仅节点、drawings仅图纸）默认mixed
                - query: 自然语言查询语句（必填）
                - chunk_weight: 文本块权重(0.0-1.0) 默认0.4
                - node_weight: 节点属性权重(0.0-1.0) 默认0.6
                - limit: 返回结果数量限制 默认10
                
        Yields:
            查询结果消息
        """
        search_type = tool_parameters.get('search_type', 'mixed')
        query: str = tool_parameters.get('query', '').strip()
        if not query:
            yield self.create_json_message({
                "status": "error",
                "message": "查询内容不能为空，请提供有效的查询文本",
                "results": ""
            })
            return

        # 获取加权参数（安全处理None值）
        chunk_weight_param = tool_parameters.get('chunk_weight')
        node_weight_param = tool_parameters.get('node_weight')
        
        chunk_weight = float(chunk_weight_param) if chunk_weight_param is not None else 0.4
        node_weight = float(node_weight_param) if node_weight_param is not None else 0.6
        limit = int(tool_parameters.get('limit', 10))

        # 验证权重（仅在混合搜索时验证）
        if search_type == 'mixed' and not (0 <= chunk_weight <= 1 and 0 <= node_weight <= 1):
            yield self.create_json_message({
                "status": "error",
                "message": "权重值必须在0-1之间",
                "results": ""
            })
            return

        store = None
        try:
            # 生成查询向量
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                yield self.create_json_message({
                    "status": "error",
                    "message": "无法生成查询向量，请检查嵌入模型是否正常",
                    "results":""
                })
                return

            # 获取store实例并根据搜索类型执行查询
            store = self._get_store()
            if search_type == 'mixed':
                # 混合搜索：文本块+节点属性
                results = store.similarity_search(
                    query_embedding=query_embedding,
                    limit=limit,
                    chunk_weight=chunk_weight,
                    node_weight=node_weight
                )
                message = f"双索引查询完成，Chunk权重: {chunk_weight}, Node权重: {node_weight}"
            elif search_type == 'nodes':
                # 仅节点属性搜索
                results = store.search_nodes(query_embedding, limit)
                message = "节点属性查询完成"
            elif search_type == 'drawings':
                # 仅图纸内容搜索
                results = store.search_chunks(query_embedding, limit)
                message = "图纸内容查询完成"
            else:
                yield self.create_json_message({
                    "status": "error",
                    "message": f"不支持的搜索类型: {search_type}",
                    "results": ""
                })
                return



            # 格式化结果
            formatted_results = self._format_results(results)

            yield self.create_json_message({
                "status": "success",
                "message": message,
                "results": formatted_results
            })

        except Exception as e:
            yield self.create_json_message({
                "status": "error",
                "message": f"查询失败: {str(e)}",
                "results": ""
            })
        finally:
            # 确保连接关闭
            if store is not None:
                try:
                    store.close()
                except Exception as close_error:
                    print(f"关闭数据库连接时发生错误: {close_error}")
                    
    def _generate_embedding(self, text: str) -> Optional[list[float]]:
        """
        生成文本的向量嵌入
        
        Args:
            text: 待处理文本
            
        Returns:
            文本向量或None
            
        Raises:
            RuntimeError: 当向量生成失败时抛出
        """
        try:
            embedder = self._get_embedder()
            if hasattr(embedder, 'embed_query'):
                embedding = embedder.embed_query(text)
            elif hasattr(embedder, 'embed_texts'):
                embeddings = embedder.embed_texts([text])
                embedding = embeddings[0] if embeddings else None
            else:
                raise RuntimeError("嵌入器必须实现embed_query或embed_texts方法")

            # 新增：检查向量有效性
            if not embedding:
                raise RuntimeError("生成的向量为空")
            if len(embedding) == 0:
                raise RuntimeError("生成的向量维度为0")

            return embedding

        except Exception as e:
            raise RuntimeError(f"向量生成失败: {str(e)}")

    @staticmethod
    def _format_results(results: list[dict]) -> list[dict]:
        """
        格式化查询结果，过滤掉properties_embedding字段
        
        Args:
            results: 原始查询结果列表
            
        Returns:
            格式化后的结果列表
        """
        formatted = []
        for res in results:
            # 过滤掉 properties_embedding
            raw_props = res.get("all_properties", {}) or {}
            filtered_props = {k: v for k, v in raw_props.items() if k != "properties_embedding"}

            formatted.append({
                "uid": res.get("uid", ""),
                "text": res.get("text", ""),
                "similarity": res.get("similarity", 0.0),
                "weighted_similarity": res.get("weighted_similarity", res.get("similarity", 0.0)),
                "source": res.get("source", "unknown"),
                "label": res.get("label", "unknown"),
                "properties": filtered_props  # 过滤后的属性
            })

        return formatted
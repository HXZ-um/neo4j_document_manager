"""
文档存储工具
将文档内容进行分块并向量化，创建Chunk节点存储原文本和向量内容，关联Drawing节点
"""

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils.neo4j_store import Neo4jVectorStore
from utils.document_processor import DocumentProcessor
from utils.vector_embedder import BgeLargeEmbedder
from _assets.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    BGE_LARGE_EMBEDDING_MODEL,
)


class Neo4jDocStoreTool(Tool):
    """
    Neo4j文档存储工具
    将文档内容进行分块并向量化，创建Chunk节点存储原文本和向量内容，关联Drawing节点
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        执行文档分块向量化并存储到Neo4j
        
        Args:
            tool_parameters: 工具参数
                - source_url: 文档源URL或路径（必填）
                - file_name: 文件名（必填）
                - text: 文本内容（必填）
                
        Yields:
            执行结果消息
        """
        neo4j_store = None
        
        try:
            # 获取参数
            # uid = tool_parameters.get('uid', '')
            source_url = tool_parameters.get('source_url', '')
            file_name = tool_parameters.get('file_name', '')
            text = tool_parameters.get('text', '')
            
            # 获取运行时配置
            runtime_config = self.runtime.credentials if hasattr(self.runtime, 'credentials') else {}
            neo4j_uri = runtime_config.get('neo4j_uri', '')
            neo4j_user = runtime_config.get('neo4j_user', '')
            neo4j_password = runtime_config.get('neo4j_password', '')

            if not all([source_url, file_name, text]):
                yield self.create_json_message({
                    "status": "error",
                    "message": "错误：参数不完整，必须包含 source_url, file_name, text",
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
            
            # 确保约束和索引存在
            neo4j_store.create_constraints()

            # 初始化向量化器
            embedder = BgeLargeEmbedder(
                model_name=BGE_LARGE_EMBEDDING_MODEL,
                use_gpu=False,
                max_batch_size=32,
                lazy_load=True
            )

            # 初始化文档处理器
            processor = DocumentProcessor(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )

            # 文本分块
            chunks_text = processor.process_text(text)

            if not chunks_text:
                yield self.create_json_message({
                    "status": "error",
                    "message": f"错误：文件 {file_name} 处理失败，未生成任何文本块"
                })
                return

            # 批量向量化
            embeddings = embedder.embed_texts(chunks_text)

            # 构建块数据
            chunks = []
            for idx, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings), start=1):
                chunks.append({
                    "id": f"{file_name}_chunk{idx}",
                    "text": chunk_text,
                    "embedding": embedding,
                    "chunk_num": idx
                })

            # 插入数据库
            success = neo4j_store.insert_pdf(
                source_url=source_url,
                fileName=file_name,
                chunks=chunks
            )

            if success:
                yield self.create_json_message({
                    "status": "success",
                    "message": f"文件 {file_name} 处理成功，共插入 {len(chunks)} 个文本块"
                })
            else:
                yield self.create_json_message({
                    "status": "error",
                    "message": f"错误：文件 '{file_name}' 插入Neo4j数据库失败"
                })

        except Exception as e:
            yield self.create_json_message({
                "status": "error",
                "message": f"执行失败：{str(e)}"
            })

        finally:
            if neo4j_store:
                neo4j_store.close()
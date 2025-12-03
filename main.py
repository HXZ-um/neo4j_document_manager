"""
FastAPI后端服务主文件
将Neo4j文档管理功能包装为独立的Web服务
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from utils.neo4j_store import Neo4jVectorStore
from utils.document_processor import DocumentProcessor
from utils.vector_embedder import BgeLargeEmbedder
from _assets.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    BGE_LARGE_EMBEDDING_MODEL,
    BGE_LARGE_LOCAL_PATH,
)

# 初始化FastAPI应用
app = FastAPI(
    title="Neo4j Document Manager API",
    description="基于Neo4j的文档管理服务，提供文档存储、语义查询和节点属性向量化功能",
    version="1.0.0"
)

# 全局变量存储服务实例
vector_store: Optional[Neo4jVectorStore] = None
embedder: Optional[BgeLargeEmbedder] = None
processor: Optional[DocumentProcessor] = None


class DatabaseConfig(BaseModel):
    """数据库配置模型"""
    uri: str
    user: str
    password: str


class DocumentStoreRequest(BaseModel):
    """文档存储请求模型"""
    source_url: str
    file_name: str
    text: str


class SemanticQueryRequest(BaseModel):
    """语义查询请求模型"""
    query_text: str
    search_type: str = "mixed"  # mixed, chunks, nodes
    top_k: int = 10
    chunk_weight: float = 0.7
    properties_weight: float = 0.3


class NodePropertiesRequest(BaseModel):
    """节点属性向量化请求模型"""
    label: str
    metadata_json: Dict[str, Any] = Field(
        ...,
        description="节点的属性信息，以JSON对象形式传入",
        example={
            "name": "设备名称",
            "type": "设备类型",
            "spec": "技术规格"
        }
    )


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global embedder, processor
    logger.info("应用启动中...")

    # ★ 使用本地 BGE 模型初始化 embedder
    embedder = BgeLargeEmbedder(
        model_name=BGE_LARGE_EMBEDDING_MODEL,   # 只用于日志展示
        local_model_path=BGE_LARGE_LOCAL_PATH,  # ★ 强制本地快照目录
        use_gpu=False,                          # 你无 GPU 就 False；有则 True
        max_batch_size=32,
        lazy_load=False
    )

    processor = DocumentProcessor(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    logger.info("应用启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global vector_store
    if vector_store:
        vector_store.close()
    logger.info("应用已关闭")


@app.post("/configure", summary="配置数据库连接")
async def configure_database(config: DatabaseConfig):
    """配置数据库连接信息"""
    global vector_store
    try:
        if vector_store:
            vector_store.close()

        vector_store = Neo4jVectorStore(
            uri=config.uri,
            user=config.user,
            password=config.password,
        )

        vector_store.create_constraints()

        return {"status": "success", "message": "数据库连接配置成功"}
    except Exception as e:
        logger.error(f"数据库连接配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"数据库连接配置失败: {str(e)}")


@app.post("/store_document", summary="存储文档")
async def store_document(request: DocumentStoreRequest):
    """将文档内容进行分块并向量化，存储到Neo4j"""
    global vector_store, embedder, processor

    if not vector_store:
        raise HTTPException(status_code=400, detail="请先配置数据库连接")

    if not embedder or not processor:
        raise HTTPException(status_code=500, detail="服务未正确初始化")

    try:
        chunks_text = processor.process_text(request.text)
        if not chunks_text:
            raise HTTPException(status_code=400, detail=f"文件 {request.file_name} 处理失败，未生成任何文本块")

        embeddings = embedder.embed_texts(chunks_text)

        chunks = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings), start=1):
            chunks.append({
                "id": f"{request.file_name}_chunk{idx}",
                "text": chunk_text,
                "embedding": embedding,
                "chunk_num": idx
            })

        success = vector_store.insert_pdf(
            source_url=request.source_url,
            fileName=request.file_name,
            chunks=chunks
        )

        if success:
            return {
                "status": "success",
                "message": f"文件 {request.file_name} 处理成功，共插入 {len(chunks)} 个文本块"
            }
        else:
            raise HTTPException(status_code=500, detail=f"文件 '{request.file_name}' 插入Neo4j数据库失败")

    except Exception as e:
        logger.error(f"文档存储失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档存储失败: {str(e)}")


@app.post("/semantic_query", summary="语义查询")
async def semantic_query(request: SemanticQueryRequest):
    """执行语义查询"""
    global vector_store, embedder

    if not vector_store:
        raise HTTPException(status_code=400, detail="请先配置数据库连接")

    if not embedder:
        raise HTTPException(status_code=500, detail="服务未正确初始化")

    try:
        query_embedding = embedder.embed_query(request.query_text)

        if request.search_type == 'chunks':
            results = vector_store.search_chunks(query_embedding, request.top_k)
            message = f"文本块查询成功，找到 {len(results)} 条相关结果"
        elif request.search_type == 'nodes':
            results = vector_store.search_nodes(query_embedding, request.top_k)
            message = f"节点查询成功，找到 {len(results)} 条相关结果"
        else:
            results = vector_store.similarity_search(
                query_embedding=query_embedding,
                limit=request.top_k,
                chunk_weight=request.chunk_weight,
                node_weight=request.properties_weight
            )
            message = f"混合查询成功，找到 {len(results)} 条相关结果"

        return {
            "status": "success",
            "results": results,
            "message": message
        }

    except Exception as e:
        logger.error(f"语义查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语义查询失败: {str(e)}")


@app.post("/node_properties", summary="节点属性向量化")
async def node_properties(request: NodePropertiesRequest):
    """将节点属性序列化后生成向量嵌入并存储到Neo4j"""
    global vector_store, embedder

    if not vector_store:
        raise HTTPException(status_code=400, detail="请先配置数据库连接")

    if not embedder:
        raise HTTPException(status_code=500, detail="服务未正确初始化")

    try:
        props = request.metadata_json

        parts = [f"label: {request.label}"]
        for k, v in props.items():
            val_str = str(v) if not isinstance(v, (dict, list)) else str(v)
            parts.append(f"{k}: {val_str}")
        serialized = " | ".join(parts)

        embedding = embedder.embed_query(serialized)

        success = vector_store.insert_node(
            label=request.label,
            props=props,
            embedding=embedding
        )

        if success:
            return {
                "status": "success",
                "message": f"节点 {request.label} 属性已成功向量化存储"
            }
        else:
            raise HTTPException(status_code=500, detail=f"节点 {request.label} 向量化存储失败")

    except Exception as e:
        logger.error(f"节点属性处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"节点属性处理失败: {str(e)}")


@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "服务运行正常"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=10086)

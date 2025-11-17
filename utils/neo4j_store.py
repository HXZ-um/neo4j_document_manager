"""
Neo4j向量存储模块
提供基于Neo4j图数据库的向量存储和检索功能，支持文档块和节点属性的双向量索引
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase, exceptions
import hashlib
import datetime


class Neo4jVectorStore:
    """Neo4j向量存储类，支持根据文件名判断文件是否存在并处理相关操作"""

    def __init__(self, uri: str, user: str, password: str, vector_dim: int = 1024):
        """
        初始化Neo4j向量存储
        
        Args:
            uri: Neo4j数据库URI
            user: 数据库用户名
            password: 数据库密码
            vector_dim: 向量维度，默认1024（BGE-large模型）
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.vector_dim = vector_dim
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        return datetime.datetime.now().isoformat()
    
    def _generate_uid(self, fileName: str) -> str:
        """
        根据文件名生成唯一ID
        
        Args:
            fileName: 文件名
            
        Returns:
            SHA256哈希值
        """
        return hashlib.sha256(fileName.encode('utf-8')).hexdigest()
    
    def _get_available_labels(self, session) -> set:
        """
        获取所有有properties_embedding的标签
        
        Args:
            session: Neo4j会话
            
        Returns:
            包含标签的集合
        """
        labels_res = session.run("""
            MATCH (n) 
            WHERE n.properties_embedding IS NOT NULL 
            RETURN DISTINCT labels(n)[0] AS label 
            LIMIT 20
        """)
        return {record["label"] for record in labels_res if record["label"]}

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        """创建约束和向量索引"""
        with self.driver.session() as session:
            # 创建唯一约束
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Drawing) REQUIRE p.fileName IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Drawing) REQUIRE p.uid IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"创建约束失败: {e}")

            # 创建Chunk向量索引
            self._create_vector_index(session, "chunk_embedding_index", "Chunk", "embedding")
    
    def _create_vector_index(self, session, index_name: str, label: str, property_name: str) -> bool:
        """
        创建向量索引的通用方法
        
        Args:
            session: Neo4j会话
            index_name: 索引名称
            label: 节点标签
            property_name: 属性名称
            
        Returns:
            创建成功返回True，否则返回False
        """
        query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON (n.{property_name})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """
        try:
            session.run(query, dim=self.vector_dim)
            return True
        except exceptions.Neo4jError as e:
            print(f"创建向量索引 {index_name} 失败: {e}")
            return False

    def _file_exists(self, uid: str, fileName: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            uid: 文件唯一ID
            fileName: 文件名
            
        Returns:
            存在返回True，否则返回False
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Drawing {uid: $uid, fileName: $fileName}) RETURN p LIMIT 1",
                uid=uid, fileName=fileName
            )
            return bool(result.single())

    def insert_pdf(self, source_url: str, fileName: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        插入PDF数据（确保事务原子性，先删除旧Chunk，再插入新Chunk）
        
        Args:
            source_url: 源URL或路径
            fileName: 文件名
            chunks: 文本块列表，每个块包含id、text、embedding、chunk_num
            
        Returns:
            插入成功返回True，否则返回False
        """
        try:
            uid = self._generate_uid(fileName)
            current_time = self._get_current_time()

            with self.driver.session() as session:
                def _tx_work(tx):
                    # 1. 合并 Drawing 节点
                    tx.run("""
                        MERGE (p:Drawing {uid: $uid, fileName: $fileName})
                        ON CREATE SET 
                            p.filePath = $filePath,
                            p.created_at = $current_time,
                            p.updated_at = $current_time
                        ON MATCH SET  
                            p.filePath = $filePath,
                            p.updated_at = $current_time
                    """, uid=uid, fileName=fileName, filePath=source_url, current_time=current_time)

                    # 2. 删除旧 Chunk
                    result = tx.run("""
                        MATCH (p:Drawing {uid: $uid, fileName: $fileName})
                        OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(p)
                        WITH COLLECT(c) AS chunks
                        FOREACH (c IN chunks | DETACH DELETE c)
                        RETURN SIZE(chunks) AS deleted_count
                    """, uid=uid, fileName=fileName)
                    deleted_count = result.single()["deleted_count"]

                    print(f"文件 {fileName} 已清理 {deleted_count} 个旧Chunk")

                    # 3. 批量插入新 Chunk
                    chunk_params = [{
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "embedding": list(chunk["embedding"]),
                        "chunk_num": chunk["chunk_num"],
                        "uid": uid,
                        "fileName": fileName,
                        "current_time": current_time
                    } for chunk in chunks]
                    
                    tx.run("""
                        UNWIND $chunks AS chunk
                        CREATE (c:Chunk {
                            id: chunk.chunk_id,
                            text: chunk.text,
                            embedding: chunk.embedding,
                            chunk_num: chunk.chunk_num,
                            created_at: chunk.current_time,
                            updated_at: chunk.current_time
                        })
                        WITH c, chunk
                        MATCH (p:Drawing {uid: chunk.uid, fileName: chunk.fileName})
                        MERGE (c)-[:PART_OF]->(p)
                    """, chunks=chunk_params)

                session.execute_write(_tx_work)

            print(f"文件 {fileName} 处理成功，共插入 {len(chunks)} 个新Chunk")
            return True

        except exceptions.Neo4jError as e:
            print(f"插入数据失败，事务已回滚: {str(e)}")
            return False



    def insert_node(self, label: str, props: Dict[str, Any], embedding: List[float]) -> bool:
        """
        插入或更新节点，并存储其属性的向量嵌入
        自动为指定标签创建向量索引（如果不存在）
        
        Args:
            label: 节点标签
            props: 节点属性字典
            embedding: 属性向量
            
        Returns:
            插入成功返回True，否则返回False
        """
        with self.driver.session() as session:
            current_time = self._get_current_time()
            
            # 为该标签创建向量索引
            index_name = f"{label.lower()}_properties_embedding_index"
            self._create_vector_index(session, index_name, label, "properties_embedding")
            
            # 构造动态MERGE语句
            set_clauses = []
            params = {
                'properties_embedding': list(embedding),
                'current_time': current_time
            }
            
            for key, value in props.items():
                set_clauses.append(f"n.{key} = ${key}")
                params[key] = value
            
            # 确定唯一性属性
            unique_key = next((key for key in ['uid', 'name', 'fileName'] if key in props), None)
            
            if unique_key:
                query = f"""
                    MERGE (n:{label} {{{unique_key}: ${unique_key}}})
                    ON CREATE SET 
                        {', '.join(set_clauses)},
                        n.properties_embedding = $properties_embedding,
                        n.created_at = $current_time,
                        n.updated_at = $current_time
                    ON MATCH SET 
                        {', '.join(set_clauses)},
                        n.properties_embedding = $properties_embedding,
                        n.updated_at = $current_time
                    RETURN n
                """
            else:
                query = f"""
                    CREATE (n:{label})
                    SET {', '.join(set_clauses)},
                        n.properties_embedding = $properties_embedding,
                        n.created_at = $current_time,
                        n.updated_at = $current_time
                    RETURN n
                """
            
            try:
                result = session.run(query, params)
                success = bool(result.single())
                print(f"节点 {label} 向量化存储{'成功' if success else '失败'}")
                return success
            except exceptions.Neo4jError as e:
                print(f"节点 {label} 向量化存储失败: {e}")
                return False

    def similarity_search(self, query_embedding: List[float], limit: int = 10, 
                         chunk_weight: float = 0.4, node_weight: float = 0.6) -> List[Dict]:
        """
        双索引向量相似度搜索，支持加权合并

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            chunk_weight: 文本块权重
            node_weight: 节点属性权重
            
        Returns:
            搜索结果列表
        """
        try:
            chunk_results = self.search_chunks(query_embedding, limit * 2)
            node_results = self.search_nodes(query_embedding, limit)
            
            # 加权处理和合并
            weighted_results = []
        
            for result in chunk_results:
                weighted_results.append({
                    **result,
                    "weighted_similarity": result["similarity"] * chunk_weight
                })

            for result in node_results:
                weighted_results.append({
                    **result,
                    "weighted_similarity": result["similarity"] * node_weight
                })

            # 排序并返回
            return sorted(weighted_results, key=lambda x: x["weighted_similarity"], reverse=True)[:limit]

        except Exception as e:
            print(f"Similarity search error: {str(e)}")
            return []

    def search_chunks(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """
        根据内容（Chunk.embedding）检索图纸
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            
        Returns:
            搜索结果列表
        """
        with self.driver.session() as session:
            try:
                res = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding)
                    YIELD node AS c, score
                    MATCH (c)-[:PART_OF]->(p:Drawing)
                    RETURN 
                        c.text AS text,
                        p.uid AS uid,
                        PROPERTIES(p) AS all_properties,
                        score AS similarity,
                        'chunk' AS source,
                        [lab IN labels(p) | lab][0] AS label
                """, query_embedding=query_embedding, top_k=limit)
                return res.data()
            except Exception as e:
                print(f"Chunk 向量查询失败: {e}")
                return []

    def search_nodes(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """
        根据属性（Node.properties_embedding）检索节点
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            
        Returns:
            搜索结果列表
        """
        results = []
        with self.driver.session() as session:
            try:
                available_labels = self._get_available_labels(session)
                print(available_labels)

                for label in available_labels:
                    index_name = f"{label.lower()}_properties_embedding_index"
                    try:
                        query_text = f"""
                            CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
                            YIELD node AS n, score
                            RETURN 
                                '' AS text,
                                COALESCE(n.uid, n.name, n.fileName, 'unknown') AS uid,
                                PROPERTIES(n) AS all_properties,
                                score AS similarity,
                                'node' AS source,
                                '{label}' AS label
                        """
                        res = session.run(query_text, query_embedding=query_embedding, top_k=limit)
                        results.extend(res.data())
                    except Exception as label_e:
                        print(f"标签 {label} 向量查询失败: {label_e}")
            except Exception as e:
                print(f"Node 向量查询失败: {e}")
        
        # 按相似度排序（降序）并限制结果数量
        sorted_results = sorted(results, key=lambda x: x.get('similarity', 0.0), reverse=True)
        return sorted_results[:limit]
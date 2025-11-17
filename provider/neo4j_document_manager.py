from typing import Any
from neo4j import GraphDatabase
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class Neo4jDocumentManagerProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            # 获取凭证
            uri = credentials.get('neo4j_uri', '')
            user = credentials.get('neo4j_user', '')
            password = credentials.get('neo4j_password', '')
            
            # 检查必需的凭证
            if not all([uri, user, password]):
                raise ToolProviderCredentialValidationError("Missing required credentials: Neo4j URI, username, and password are required.")
            
            # 确保类型正确
            uri = str(uri)
            user = str(user)
            password = str(password)
            
            # 尝试连接到Neo4j数据库以验证凭证
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            
        except ToolProviderCredentialValidationError:
            raise
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"Failed to connect to Neo4j database: {str(e)}")
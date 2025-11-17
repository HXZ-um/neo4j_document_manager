"""
文档处理器模块
负责接收纯文本内容，执行清洗和智能分块处理
"""

import re
from typing import List
from _assets.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class DocumentProcessor:
    """文档处理类，负责接收纯文本内容，执行清洗和分割"""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        初始化处理器，支持自定义文本块大小和重叠度
        
        Args:
            chunk_size: 单个文本块的最大长度（超过则拆分），默认使用配置文件值
            chunk_overlap: 相邻文本块的重叠长度（当前用于长文本强制拆分时的补偿，备用扩展）
        """
        # 校验参数合法性，避免无效值
        self.chunk_size = max(10, chunk_size)  # 最小10字符，防止过小导致拆分碎片化
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))  # 重叠度不超过块大小的1/2

    def clean_text(self, text: str) -> str:
        """
        轻量清洗：仅规范空白，保留原始标点与换行
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
            
        Raises:
            TypeError: 当输入不是字符串类型时抛出
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串类型")

        # 1. 替换连续制表符/回车为空格，保留换行符
        text = re.sub(r'[ \t\r]+', ' ', text)

        # 2. 去除整体首尾多余空白，不影响行内排版
        text = text.strip()

        # 3. 合并过多的连续换行（最多保留 2 个换行，避免大面积空行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def split_text(self, text: str) -> List[str]:
        """
        分层分割逻辑（优先保持语义完整性）：
        1. 按段落（\n\n）拆分，优先保留完整段落
        2. 若段落超过 chunk_size，按单行（\n）拆分，保留完整行
        3. 若单行仍超过 chunk_size，强制按 chunk_size 拆分（带重叠补偿）
        
        Args:
            text: 待分割的文本
            
        Returns:
            分割后的文本块列表
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        chunks: List[str] = []
        current_chunk_size = self.chunk_size  # 引用类属性，避免变量未定义

        # 第一步：按段落拆分（\n\n 分隔，语义最完整）
        paragraphs = cleaned_text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue  # 跳过空段落

            # 段落长度合适，直接作为一个块
            if len(para) <= current_chunk_size:
                chunks.append(para)
                continue

            # 第二步：段落过长，按单行拆分（\n 分隔，保留行级语义）
            lines = para.split("\n")
            line_buffer = ""  # 缓冲区存储待合并的行
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 计算当前缓冲区+新行的长度（+1 是换行符的占位）
                combined_length = len(line_buffer) + len(line) + 1
                if combined_length <= current_chunk_size:
                    # 缓冲区可容纳，添加到缓冲区（用换行符连接）
                    line_buffer = f"{line_buffer}\n{line}" if line_buffer else line
                else:
                    # 缓冲区满，添加到结果，重置缓冲区
                    if line_buffer:
                        chunks.append(line_buffer)
                    # 检查单行是否超过块大小（极端情况：单行超长）
                    if len(line) > current_chunk_size:
                        # 第三步：单行超长，强制拆分（带重叠，避免语义断裂）
                        chunks.extend(self._force_split_long_text(line, current_chunk_size))
                    else:
                        # 单行长度合适，直接作为缓冲区初始值
                        line_buffer = line

            # 处理单行循环结束后的剩余缓冲区
            if line_buffer:
                chunks.append(line_buffer)

        return chunks

    def _force_split_long_text(self, text: str, max_length: int) -> List[str]:
        """
        辅助方法：强制拆分超长文本（单行超过 chunk_size 时使用）
        带重叠补偿：相邻块重叠 chunk_overlap 字符，避免拆分导致语义断裂
        
        Args:
            text: 超长文本
            max_length: 最大长度
            
        Returns:
            拆分后的文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算当前块的结束位置（最后一块不强制长度）
            end = min(start + max_length, text_length)
            # 提取当前块
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # 更新下一块的起始位置（带重叠）
            start = end - self.chunk_overlap if (end < text_length) else text_length

        return chunks

    def process_text(self, text: str) -> List[str]:
        """
        一步完成"清洗→分割"，对外提供统一接口
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本块列表
        """
        try:
            return self.split_text(text)
        except Exception as e:
            # 捕获异常并友好提示，避免程序崩溃
            print(f"文档处理失败：{str(e)}")
            return []
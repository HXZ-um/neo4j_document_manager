"""
BGE向量嵌入器模块
提供基于BAAI/bge-large-zh-v1.5模型的文本向量化功能
"""

import logging
import threading
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEmbedder:
    """嵌入器基类"""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量列表
        """
        raise NotImplementedError("子类必须实现此方法")

    def embed_query(self, query: str) -> List[float]:
        """
        生成单个查询文本的向量
        
        Args:
            query: 查询文本
            
        Returns:
            查询文本向量
        """
        return self.embed_texts([query])[0]


class BgeLargeEmbedder(BaseEmbedder):
    """
    BGE大型嵌入器类
    基于BAAI/bge-large-zh-v1.5模型实现文本向量化功能，支持GPU加速和线程安全
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        max_batch_size: int = 64,
        max_length: Optional[int] = 512,
        pooling: str = "mean",  # "mean" or "cls"
        dtype: Optional[torch.dtype] = None,  # e.g. torch.float16 for GPU
        trust_remote_code: bool = False,
        lazy_load: bool = False,
    ):
        """
        初始化BGE嵌入器
        
        Args:
            model_name: 模型名称（如"BAAI/bge-large-zh-v1.5"）
            use_gpu: 是否使用GPU加速
            max_batch_size: 最大批处理大小
            max_length: 最大文本长度
            pooling: 向量池化方式（"mean"或"cls"）
            dtype: 数据类型（如torch.float16用于GPU混合精度）
            trust_remote_code: 是否信任远程代码
            lazy_load: 是否启用懒加载
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.max_batch_size = max_batch_size
        self.max_length = max_length or 512
        self.pooling = pooling
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        # tokenizer 可以立即加载（通常开销小），model 支持 lazy load
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._model = None
        self._model_lock = threading.Lock()  # 保护模型加载与推理
        self._inference_lock = threading.Lock()  # 可选：串行化推理以避免某些并发问题
        self.retrieval_instruction = "为这个句子生成表示以用于检索相关文本："

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """
        加载模型到指定设备
        使用线程锁确保线程安全
        """
        with self._model_lock:
            if self._model is not None:
                return
            logger.info(f"加载模型 {self.model_name} 到 {self.device}，dtype={self.dtype} ...")
            try:
                kwargs = {}
                if self.dtype is not None:
                    kwargs["torch_dtype"] = self.dtype
                # from_pretrained 接受 torch_dtype（如果支持）
                self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code, **kwargs)
                self._model.to(self.device)
                self._model.eval()
                logger.info("模型加载完成")
            except Exception as e:
                logger.exception("模型加载失败")
                raise RuntimeError(f"模型加载失败: {e}") from e

    def _pool_embeddings(self, token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        对标记嵌入进行池化处理
        
        Args:
            token_embeddings: 标记嵌入张量 (batch, seq_len, dim)
            attention_mask: 注意力掩码
            
        Returns:
            池化后的嵌入张量
        """
        # token_embeddings: (batch, seq_len, dim)
        if self.pooling == "cls":
            return token_embeddings[:, 0, :]
        # mean pooling with attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)  # (batch, seq_len, 1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1e-9)
            return summed / counts
        # fallback: mean over tokens
        return token_embeddings.mean(dim=1)

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量列表
            
        Raises:
            RuntimeError: 当模型加载或推理失败时抛出
        """
        if not texts:
            return []

        # lazy load model if needed
        if self._model is None:
            self._load_model()

        embeddings_out = []
        try:
            # 分批处理，防止 OOM
            for i in range(0, len(texts), self.max_batch_size):
                batch_texts = texts[i: i + self.max_batch_size]
                input_texts = [f"{self.retrieval_instruction}{t}" for t in batch_texts]

                inputs = self.tokenizer(
                    input_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                with self._inference_lock:
                    with torch.no_grad():
                        # 确保模型已加载
                        if self._model is None:
                            raise RuntimeError("模型未加载，无法进行推理")
                        
                        # 在 GPU 上使用 autocast（自动混合精度）提升速度并节省显存
                        if self.device.type == "cuda":
                            with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16 or self.dtype is None)):
                                outputs = self._model(**inputs)
                        else:
                            outputs = self._model(**inputs)

                # 兼容不同类型的返回（namedtuple / BaseModelOutput / tuple）
                token_embeddings = getattr(outputs, "last_hidden_state", None)
                if token_embeddings is None:
                    # outputs[0] is generally token embeddings
                    token_embeddings = outputs[0]

                # pooling（prefer mean with mask）
                attention_mask = inputs.get("attention_mask", None)
                pooled = self._pool_embeddings(token_embeddings, attention_mask)
                normalized = F.normalize(pooled, p=2, dim=1)

                # 转回 CPU 并转成 python list
                embeddings_out.extend(normalized.cpu().numpy().tolist())

            return embeddings_out

        except Exception as e:
            logger.exception("生成嵌入失败")
            raise RuntimeError(f"BGE-large 批量嵌入失败 ({self.device}): {e}") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量列表
        """
        if not texts:
            return []
        logger.debug(f"生成 {len(texts)} 条文本的嵌入（分批最大 {self.max_batch_size}）")
        embeddings = self._generate_embeddings_batch(texts)
        if embeddings:
            logger.info(f"生成完成，共 {len(embeddings)} 个向量，向量维度 {len(embeddings[0])}")
        else:
            logger.warning("未生成任何向量")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        生成单个查询文本的向量
        
        Args:
            query: 查询文本
            
        Returns:
            查询文本向量
            
        Raises:
            ValueError: 当查询文本为空时抛出
            RuntimeError: 当向量生成失败时抛出
        """
        if query is None or not query.strip():
            raise ValueError("查询文本不能为空")
        try:
            emb = self._generate_embeddings_batch([query])
            if not emb:
                raise RuntimeError("没有生成向量")
            return emb[0]
        except Exception as e:
            raise RuntimeError(f"查询向量生成失败: {e}") from e
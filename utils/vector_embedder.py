"""
BGE向量嵌入器模块
提供基于BAAI/bge-large-zh-v1.5模型的文本向量化功能（强制本地加载版）
CPU / GPU 通用：use_gpu=True 且 CUDA 可用时走 GPU，否则自动回退 CPU
"""

import logging
import threading
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEmbedder:
    """嵌入器基类"""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
        raise NotImplementedError("子类必须实现此方法")

    def embed_query(self, query: str) -> List[float]:
        """生成单个查询文本的向量"""
        return self.embed_texts([query])[0]


class BgeLargeEmbedder(BaseEmbedder):
    """
    BGE大型嵌入器类（强制本地加载）
    基于BAAI/bge-large-zh-v1.5模型实现文本向量化功能，支持GPU加速和线程安全
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",   # 仅用于日志/展示
        local_model_path: Optional[str] = None,     # 本地快照目录（snapshots/<hash>）
        use_gpu: bool = False,
        max_batch_size: int = 64,
        max_length: Optional[int] = 512,
        pooling: str = "mean",  # "mean" or "cls"
        dtype: Optional[torch.dtype] = None,  # GPU 可用 torch.float16
        trust_remote_code: bool = False,
        lazy_load: bool = False,
    ):
        if local_model_path is None:
            raise ValueError("local_model_path 不能为空！请传入本地 snapshots/<hash> 目录。")

        self.model_name = model_name
        self.local_model_path = local_model_path
        self.use_gpu = use_gpu

        # 设备自动选择：use_gpu=True 且 CUDA 可用 -> cuda，否则 cpu
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )

        self.max_batch_size = max_batch_size
        self.max_length = max_length or 512
        self.pooling = pooling
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        # 强制从本地加载 tokenizer
        logger.info(f"从本地加载 tokenizer: {self.local_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True
        )

        self._model = None
        self._model_lock = threading.Lock()      # 保护模型加载与推理
        self._inference_lock = threading.Lock()  # 推理锁（避免某些并发问题）
        self.retrieval_instruction = "为这个句子生成表示以用于检索相关文本："

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """加载模型到指定设备（线程安全）"""
        with self._model_lock:
            if self._model is not None:
                return

            logger.info(
                f"从本地加载模型 {self.model_name} -> {self.local_model_path} "
                f"到 {self.device}，dtype={self.dtype} ..."
            )
            try:
                kwargs = {}
                if self.dtype is not None:
                    kwargs["torch_dtype"] = self.dtype

                # 强制本地加载模型
                self._model = AutoModel.from_pretrained(
                    self.local_model_path,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=True,
                    **kwargs
                )

                self._model.to(self.device)
                self._model.eval()
                logger.info("模型加载完成（本地）")

            except Exception as e:
                logger.exception("模型加载失败")
                raise RuntimeError(f"模型加载失败: {e}") from e

    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """对标记嵌入进行池化处理"""
        if self.pooling == "cls":
            return token_embeddings[:, 0, :]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1e-9)
            return summed / counts

        return token_embeddings.mean(dim=1)

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入向量"""
        if not texts:
            return []

        # lazy load
        if self._model is None:
            self._load_model()

        embeddings_out = []
        try:
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
                        if self._model is None:
                            raise RuntimeError("模型未加载，无法进行推理")

                        if self.device.type == "cuda":
                            # GPU 下混合精度
                            with torch.cuda.amp.autocast(
                                enabled=(self.dtype == torch.float16 or self.dtype is None)
                            ):
                                outputs = self._model(**inputs)
                        else:
                            outputs = self._model(**inputs)

                token_embeddings = getattr(outputs, "last_hidden_state", None)
                if token_embeddings is None:
                    token_embeddings = outputs[0]

                attention_mask = inputs.get("attention_mask", None)
                pooled = self._pool_embeddings(token_embeddings, attention_mask)
                normalized = F.normalize(pooled, p=2, dim=1)

                embeddings_out.extend(normalized.cpu().numpy().tolist())

            return embeddings_out

        except Exception as e:
            logger.exception("生成嵌入失败")
            raise RuntimeError(f"BGE-large 批量嵌入失败 ({self.device}): {e}") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本向量"""
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
        """生成单个查询文本的向量"""
        if query is None or not query.strip():
            raise ValueError("查询文本不能为空")

        emb = self._generate_embeddings_batch([query])
        if not emb:
            raise RuntimeError("没有生成向量")
        return emb[0]

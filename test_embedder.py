import numpy as np

from utils.vector_embedder import BgeLargeEmbedder
from _assets.config import BGE_LARGE_LOCAL_PATH, BGE_LARGE_EMBEDDING_MODEL

def main():
    embedder = BgeLargeEmbedder(
        model_name=BGE_LARGE_EMBEDDING_MODEL,
        local_model_path=BGE_LARGE_LOCAL_PATH,
        use_gpu=False,          # 你没GPU就False
        pooling="mean",
        lazy_load=False         # 立刻加载，便于测试
    )

    # 1) 单句向量
    q = "你好，世界"
    v = embedder.embed_query(q)
    print("single dim =", len(v))
    print("single head =", v[:5])

    # 2) 批量向量
    texts = ["今天天气不错", "我想去北京旅游", "向量检索是什么"]
    vecs = embedder.embed_texts(texts)
    print("batch size =", len(vecs))
    print("batch dim  =", len(vecs[0]))

    a = embedder.embed_query("北京今天下雨了吗？")
    b = embedder.embed_query("北京的天气怎么样？")
    c = embedder.embed_query("我想买一台显卡")

    def cos(x, y):
        x = np.array(x);
        y = np.array(y)
        return float(np.dot(x, y))

    print("cos(a,b) =", cos(a, b))  # 应该更高
    print("cos(a,c) =", cos(a, c))  # 应该更低

if __name__ == "__main__":
    main()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import faiss
import numpy as np
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAG:
    def __init__(self, dimension, use_qdrant=False, qdrant_config=None):
        """
        初始化 RAG 模块，可选择启用 Qdrant 存储。
        
        :param dimension: 向量的维度。
        :param use_qdrant: 是否启用 Qdrant 存储。
        :param qdrant_config: Qdrant 配置字典（仅在 use_qdrant=True 时有效）。
        """
        self.dimension = dimension
        self.index = {
            "text": faiss.IndexFlatL2(dimension),
            "image": faiss.IndexFlatL2(dimension),
            "audio": faiss.IndexFlatL2(dimension),
            "video": faiss.IndexFlatL2(dimension),
        }
        self.data_map = {
            "text": [],
            "image": [],
            "audio": [],
            "video": [],
        }
        self.use_qdrant = use_qdrant
        self.qdrant_client = None

        # 如果使用 Qdrant，初始化 Qdrant 客户端
        if self.use_qdrant:
            if not qdrant_config:
                raise ValueError("Qdrant configuration is required when use_qdrant is True.")
            self.qdrant_client = QdrantClient(**qdrant_config)
            self.collection_names = {
                "text": "text_collection",
                "image": "image_collection",
                "audio": "audio_collection",
                "video": "video_collection",
            }
            self._init_qdrant()

    def _init_qdrant(self):
        """
        初始化 Qdrant Collections。
        """
        logger.info("Initializing Qdrant collections...")
        for modality, collection_name in self.collection_names.items():
            if not self.qdrant_client.get_collection(collection_name, raise_on_not_found=False):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
        logger.info("Qdrant collections initialized.")

    def add_to_index(self, embeddings, data, modality, ids=None):
        """
        添加向量到 FAISS 或 Qdrant 索引。

        :param embeddings: 嵌入向量的 numpy 数组。
        :param data: 原始数据列表，与嵌入对齐。
        :param modality: 数据模态类型 ("text", "image", "audio", "video")。
        :param ids: 数据的唯一 ID 列表（可选）。
        """
        if modality not in self.index:
            raise ValueError(f"Unsupported modality: {modality}")

        if self.use_qdrant:
            # 添加到 Qdrant
            vectors = [
                PointStruct(
                    id=str(ids[i] if ids else i),
                    vector=embeddings[i].tolist(),
                    payload={"data": data[i], "modality": modality},
                )
                for i in range(len(embeddings))
            ]
            try:
                self.qdrant_client.upsert(collection_name=self.collection_names[modality], points=vectors)
                logger.info(f"Successfully added {len(embeddings)} {modality} vectors to Qdrant.")
            except Exception as e:
                logger.error(f"Failed to add {modality} vectors to Qdrant: {e}")
                raise
        else:
            # 添加到 FAISS
            self.index[modality].add(embeddings)
            self.data_map[modality].extend(data)
            logger.info(f"Successfully added {len(embeddings)} {modality} vectors to FAISS.")

    def retrieve(self, query_embedding, modality, top_k=5):
        """
        检索最相关的结果。

        :param query_embedding: 查询向量。
        :param modality: 数据模态类型 ("text", "image", "audio", "video")。
        :param top_k: 返回最相关的结果数。
        :return: 检索到的结果列表。
        """
        if modality not in self.index:
            raise ValueError(f"Unsupported modality: {modality}")

        try:
            if self.use_qdrant:
                query_result = self.qdrant_client.search(
                    collection_name=self.collection_names[modality],
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                )
                results = [res.payload["data"] for res in query_result]
            else:
                query_embedding = np.array([query_embedding]).astype("float32")
                _, indices = self.index[modality].search(query_embedding, top_k)
                results = [self.data_map[modality][i] for i in indices[0] if i < len(self.data_map[modality])]
            logger.info(f"Successfully retrieved {len(results)} {modality} results.")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve {modality} results: {e}")
            raise

   
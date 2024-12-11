from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import faiss
import numpy as np
from uuid import uuid4

class RAG:
    def __init__(self, use_qdrant=False, qdrant_config=None):
        """
        初始化 RAG 模块，可选择启用 Qdrant 存储。

        :param use_qdrant: 是否启用 Qdrant 存储。
        :param qdrant_config: Qdrant 配置字典（仅在 use_qdrant=True 时有效）。
        """
        self.use_qdrant = use_qdrant
        self.index = {}  # 动态存储每个模态的索引
        self.data_map = {}  # 动态存储每个模态的数据
        self.qdrant_client = None
        self.collection_names = {
            "text": "text_collection",
            "image": "image_collection",
            "audio": "audio_collection",
            "video": "video_collection",
        }
        self.dimensions = {}  # 动态存储每个模态的维度

        # 如果使用 Qdrant，初始化 Qdrant 客户端
        if self.use_qdrant:
            if not qdrant_config:
                raise ValueError("Qdrant configuration is required when use_qdrant is True.")
            self.qdrant_client = QdrantClient(**qdrant_config)

    def set_embedding_dimension(self, modality, embeddings):
        """
        根据给定的模态和嵌入向量，动态设置索引的维度，并创建 Qdrant 集合。

        :param modality: 数据模态 ("text", "image", "audio", "video")
        :param embeddings: 嵌入向量，形状为 [N, D]
        """
        if embeddings.ndim > 1:
            dimension = embeddings.shape[1]  # 确保获取正确的维度
        else:
            dimension = len(embeddings)

        print(f"Setting embedding dimension for {modality}: {dimension}")

        # 如果维度尚未设置，则初始化索引和 Qdrant 集合
        if modality not in self.dimensions or self.dimensions[modality] != dimension:
            self.dimensions[modality] = dimension
            self.index[modality] = faiss.IndexFlatL2(dimension)
            self.data_map[modality] = []

            if self.use_qdrant:
                collection_name = self.collection_names[modality]
                try:
                    self.qdrant_client.get_collection(collection_name)
                    print(f"Collection {collection_name} already exists. Deleting it.")
                    self.qdrant_client.delete_collection(collection_name)
                    print(f"Collection {collection_name} deleted successfully.")
                except Exception as e:
                    if "not found" in str(e).lower():
                        print(f"Collection {collection_name} does not exist, skipping deletion.")
                    else:
                        raise e

                try:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                    )
                    print(f"Qdrant collection {collection_name} created with dimension {dimension}.")
                except Exception as create_e:
                    print(f"Error creating Qdrant collection {collection_name}: {create_e}")
                    raise


    def add_to_index(self, embeddings, data, modality, ids=None):
        """
        添加向量到 FAISS 或 Qdrant 索引。

        :param embeddings: 嵌入向量的 numpy 数组。
        :param data: 原始数据列表，与嵌入对齐。
        :param modality: 数据模态类型 ("text", "image", "audio", "video")。
        :param ids: 数据的唯一 ID 列表（可选）。
        """
        # 检查索引是否已经存在
        if modality not in self.index:
            raise ValueError(f"Unsupported modality: {modality}")

        if self.use_qdrant:
            vectors = [
                PointStruct(
                    id=str(ids[i] if ids else uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={"data": data[i], "modality": modality},
                )
                for i in range(len(embeddings))
            ]
            try:
                self.qdrant_client.upsert(collection_name=self.collection_names[modality], points=vectors)
                print(f"Upserted {len(vectors)} points to Qdrant collection {modality}.")
            except Exception as e:
                print(f"Error upserting to Qdrant: {e}")
                raise
        else:
            self.index[modality].add(embeddings)
            self.data_map[modality].extend(data)
            print(f"Added {len(embeddings)} embeddings to FAISS index for {modality}.")

    def retrieve(self, query_embedding, modality, top_k=5):
        """
        检索最相关的结果。

        :param query_embedding: 查询向量。
        :param modality: 数据模态类型 ("text", "image", "audio", "video")。
        :param top_k: 返回最相关的结果数。
        :return: 检索到的结果列表。
        """
        query_dim = len(query_embedding) if isinstance(query_embedding, list) else query_embedding.shape[1]
        print(f"Query embedding dimension for {modality}: {query_dim}")

        # 检查模态是否支持
        if modality not in self.index:
            raise ValueError(f"Unsupported modality: {modality}")

        # 获取存储在索引中的维度
        stored_dim = self.dimensions.get(modality)
        if stored_dim != query_dim:
            raise ValueError(f"Query embedding dimension {query_dim} does not match stored dimension {stored_dim}.")

        try:
            if self.use_qdrant:
                collection_name = self.collection_names[modality]
                query_result = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                )
                results = [res.payload["data"] for res in query_result]
                print(f"Retrieved {len(results)} results from Qdrant for modality {modality}.")
            else:
                query_embedding = np.array([query_embedding]).astype("float32")
                _, indices = self.index[modality].search(query_embedding, top_k)
                results = [self.data_map[modality][i] for i in indices[0] if i < len(self.data_map[modality])]
                print(f"Retrieved {len(results)} results from FAISS for modality {modality}.")
            return results
        except Exception as e:
            print(f"Error during retrieval for modality {modality}: {e}")
            raise

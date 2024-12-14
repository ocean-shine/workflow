from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np

class RAG:
    def __init__(self, use_qdrant=True, qdrant_config=None):
        """
        初始化 RAG 类，用于处理向量存储和检索。

        :param use_qdrant: 是否使用 Qdrant 作为向量存储。
        :param qdrant_config: Qdrant 配置信息，用于初始化 Qdrant 客户端。
        """
        self.use_qdrant = use_qdrant
        self.qdrant_config = qdrant_config  # 将 qdrant_config 存储为实例变量
        self.qdrant_client = None
        self.embeddings_dims = {}  # 用于存储不同模态的维度

        if self.use_qdrant and qdrant_config:
            # 如果启用 Qdrant，初始化 Qdrant 客户端
            self.initialize_qdrant(qdrant_config)
        else:
            print("Qdrant client not initialized.")

    def initialize_qdrant(self, qdrant_config):
        """初始化 Qdrant 客户端"""
        try:
            # 验证 Qdrant 配置是否有效
            required_keys = ["host", "port"]
            for key in required_keys:
                if key not in qdrant_config:
                    raise ValueError(f"Missing required config key: {key}")
            
            # 初始化 Qdrant 客户端
            self.qdrant_client = QdrantClient(**qdrant_config)
            print("Qdrant client initialized successfully.")
            
            # 确保集合已清理
            self._ensure_collections_clean()
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            self.qdrant_client = None  # Ensure client is set to None if initialization fails.

    def _ensure_collections_clean(self):
        """
        检查 Qdrant 中是否已存在集合，若存在则删除。
        """
        if self.qdrant_client:
            # 遍历不同模态并删除已有的集合
            for modality in self.embeddings_dims.keys():
                collection_name = f"{modality}_collection"
                try:
                    self.qdrant_client.get_collection(collection_name)
                    print(f"Collection '{collection_name}' already exists. Deleting it...")
                    self.qdrant_client.delete_collection(collection_name)
                    print(f"Collection '{collection_name}' deleted successfully.")
                except Exception as e:
                    if "not found" in str(e).lower():
                        print(f"Collection '{collection_name}' does not exist. Skipping deletion.")
                    else:
                        raise e
        else:
            print("Qdrant client is not initialized, skipping collection cleanup.")

    def set_embedding_dimension(self, modality, embeddings):
        """自动设置嵌入维度，根据模态和嵌入向量的维度"""
        if embeddings is not None and hasattr(embeddings, 'shape'):
            dimension = embeddings.shape[1]  # 获取嵌入的维度
            self.embeddings_dims[modality] = dimension
            print(f"Embedding dimension for modality '{modality}' set to {dimension}")
        else:
            raise ValueError(f"Invalid embeddings input for modality '{modality}' or embeddings do not have shape attribute.")

    def detect_embedding_dimension(self, embeddings):
        """检测嵌入的维度"""
        # 检查嵌入的形状
        if embeddings is not None and hasattr(embeddings, 'shape'):
            dimension = embeddings.shape[1]  # 获取第二个维度作为嵌入维度
            return dimension
        else:
            print("Invalid embeddings input or embeddings do not have shape attribute.")
            return None

    def add_to_index(self, embeddings, data, modality, dimension=None):
        """将数据和嵌入添加到 Qdrant 索引"""
        if self.qdrant_client:
            # 如果没有提供维度，从已存储的维度字典中获取
            if not dimension:
                if modality not in self.embeddings_dims:
                    self.set_embedding_dimension(modality, embeddings)
                dimension = self.embeddings_dims[modality]
            
            collection_name = f"{modality}_collection"

            # 新建集合 (确保集合不存在时才新建)
            try:
                self.qdrant_client.get_collection(collection_name)
            except:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                print(f"New collection '{collection_name}' created.")

            # 添加数据到集合
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": i,
                        "vector": embedding.tolist(),
                        "payload": {"data": item}
                    }
                    for i, (embedding, item) in enumerate(zip(embeddings, data))
                ]
            )
            print(f"Added {len(embeddings)} embeddings to {collection_name} collection.")
        else:
            print("Qdrant client is not initialized, skipping adding embeddings.")
            raise ValueError("Qdrant client is not initialized.")

    def retrieve(self, query_embedding, modality, top_k=5):
        """
        根据查询嵌入检索最相关的结果

        :param query_embedding: 查询嵌入向量
        :param modality: 数据模态类型 (可以是任意字符串，如 "audio", "text", "video" 等)
        :param top_k: 返回最相关的结果数
        :return: 检索到的结果列表
        """
        if self.qdrant_client:
            collection_name = f"{modality}_collection"
            query_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            results = [res.payload["data"] for res in query_result]
            print(f"Retrieved {len(results)} results from Qdrant for modality '{modality}'.")
            return results
        else:
            print("Qdrant client is not initialized, skipping retrieval.")
            raise ValueError("Qdrant client is not initialized.")

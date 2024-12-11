from .rag import RAG
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRetrieval:
    def __init__(self):
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()
        self.rag = RAG(use_qdrant=True, qdrant_config={
            'url': 'http://localhost:6333',
            'port': 6333
        })
        # 确保 Qdrant 集合已初始化
        # self.rag._init_qdrant()  # 添加这一行来初始化 Qdrant 集合

    def retrieval_audio(self, query: str, file_path: str) -> dict:
        # 获取音频嵌入
        audio_embedding = self.audio_encoder.encode(file_path)

        # 将音频嵌入向量转换为 NumPy 数组（确保格式兼容）
        audio_embedding = np.array(audio_embedding)

        # 动态设置音频嵌入的维度（根据嵌入向量的维度）
        self.rag.set_embedding_dimension(modality="audio", embeddings=audio_embedding)

        # 添加音频嵌入到索引
        self.rag.add_to_index([audio_embedding], [file_path], modality="audio")

        # 编码查询文本
        query_embedding = self.text_encoder.encode(query)

        # 检索与查询相似的音频数据
        retrieved = self.rag.retrieve(query_embedding, modality="audio", top_k=5)

        return {"retrieved_data": retrieved}

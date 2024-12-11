from .rag import RAG
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRetrieval:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.rag = RAG(use_qdrant=True, qdrant_config={
            'url': 'http://localhost:6333',
            'port': 6333
        })
        # 确保 Qdrant 集合已初始化
        # self.rag._init_qdrant()  # 添加这一行来初始化 Qdrant 集合

    def retrieval_image(self, query: str, file_path: str) -> dict:
        from PIL import Image

        # 打开图像文件并获取图像嵌入
        image = Image.open(file_path)
        image_embedding = self.image_encoder.encode(image)

        # 将图像嵌入向量转换为 NumPy 数组（确保格式兼容）
        image_embedding = np.array(image_embedding)

        # 动态设置图像的维度（根据嵌入向量的维度）
        self.rag.set_embedding_dimension(modality="image", embeddings=image_embedding)

        # 将图像嵌入向量添加到索引
        self.rag.add_to_index([image_embedding], [file_path], modality="image")

        # 编码查询
        query_embedding = self.text_encoder.encode(query)

        # 检索图像相关数据
        retrieved = self.rag.retrieve(query_embedding, modality="image", top_k=5)

        return {"retrieved_data": retrieved}

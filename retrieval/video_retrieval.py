import logging
import os
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from shutil import rmtree
from .rag import RAG
from models.video_encoder import VideoEncoder
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRetrieval:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.video_encoder = VideoEncoder(self.image_encoder)
        self.text_encoder = TextEncoder()
        self.rag = RAG(use_qdrant=True, qdrant_config={
            'url': 'http://localhost:6333',
            'port': 6333
        })
        
        # 确保 Qdrant 集合已初始化
        self.rag.set_embedding_dimension(modality="video", embeddings=np.zeros((1, 512)))  # 使用默认值初始化维度

    def retrieval_video(self, query: str, file_path: str) -> dict:
        """
        根据查询文本和视频路径，检索与查询最相关的视频帧。

        :param query: 查询文本。
        :param file_path: 视频文件路径。
        :return: 包含检索到的视频帧信息的字典。
        """
        # 提取视频帧并存储
        video = VideoFileClip(file_path)
        frame_paths = []
        interval = 10  # 每 10 秒提取一帧

        frames_folder = "frames"
        os.makedirs(frames_folder, exist_ok=True)

        for t in range(0, int(video.duration), interval):
            frame_path = os.path.join(frames_folder, f"frame_{t:04}.png")
            video.save_frame(frame_path, t)
            frame_paths.append(frame_path)

        # 编码帧并获取嵌入向量
        frame_embeddings = self.video_encoder.encode_frames(frame_paths)

        # # 将嵌入向量转换为 NumPy 数组（确保格式兼容）
        # frame_embeddings = np.array(frame_embeddings)

        # 动态设置视频帧嵌入的维度
        self.rag.set_embedding_dimension(modality="video", embeddings=frame_embeddings)

        # 将视频帧嵌入向量添加到 Qdrant 索引
        self.rag.add_to_index(frame_embeddings, frame_paths, modality="video")

        # 清理临时帧文件
        rmtree(frames_folder)  # 删除所有帧文件

        # 编码查询文本
        query_embedding = self.text_encoder.encode(query)

        # 检索与查询相似的视频帧
        retrieved = self.rag.retrieve(query_embedding, modality="video", top_k=5)

        return {"retrieved_data": retrieved}

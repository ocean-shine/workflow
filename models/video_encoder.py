from PIL import Image
import numpy as np
from models.image_encoder import ImageEncoder

class VideoEncoder:
    def __init__(self, image_encoder: ImageEncoder):
        self.image_encoder = image_encoder

    def encode_frames(self, frame_paths: list) -> np.ndarray:
        """
        编码视频帧，返回每帧的嵌入向量。

        :param frame_paths: 视频帧的路径列表。
        :return: 每个帧的嵌入向量数组。
        """
        embeddings = []
        for frame_path in frame_paths:
            image = Image.open(frame_path)
            embedding = self.image_encoder.encode(image)
            embeddings.append(embedding)
        return np.array(embeddings)

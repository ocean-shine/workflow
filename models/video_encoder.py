import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

class VideoEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        初始化 CLIP 模型用于视频特征提取。
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _extract_frames(self, video_path, num_frames=16):
        """
        从视频中均匀抽取帧。

        :param video_path: 视频文件路径。
        :param num_frames: 要抽取的帧数。
        :return: 抽取的帧列表。
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算均匀抽取的帧索引
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为 RGB 格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def encode(self, video_path):
        """
        从视频中提取特征向量。

        :param video_path: 视频文件路径。
        :return: 视频嵌入向量。
        """
        frames = self._extract_frames(video_path)

        # 预处理帧
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            padding=True
        )

        # 提取视频嵌入
        with torch.no_grad():
            video_features = self.model.get_image_features(**inputs)

        # 对帧特征取平均值，得到视频全局表示
        video_embedding = video_features.mean(dim=0)
        return video_embedding.numpy()

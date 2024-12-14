# models/image_encoder.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class ImageEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image_path):
        """从文件路径加载图像并计算其嵌入"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings.cpu().numpy().flatten()

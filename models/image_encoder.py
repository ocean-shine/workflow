from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class ImageEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化 ImageEncoder 类，加载 CLIP 模型。

        :param model_name: 使用的 CLIP 模型的名称，默认为 "openai/clip-vit-base-patch32"。
        """
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

    def encode(self, image: Image.Image) -> np.ndarray:
        """
        对输入图像进行编码，获取其嵌入向量。

        :param image: 输入的 PIL Image 对象。
        :return: 图像的嵌入向量（NumPy 数组）。
        """
        try:
            inputs = self.processor(images=image, return_tensors='pt')
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
            # 将嵌入向量从 PyTorch tensor 转换为 NumPy 数组
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            raise

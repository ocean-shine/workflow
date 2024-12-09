from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class ImageEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        初始化 CLIP 模型用于图像特征提取。
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, image_path):
        """
        从图像文件中提取特征向量。

        :param image_path: 图像文件路径。
        :return: 图像嵌入向量。
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.squeeze(0).numpy()

    def encode_text(self, text):
        """
        从文本中提取嵌入，用于查询。

        :param text: 查询文本。
        :return: 文本嵌入向量。
        """
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.squeeze(0).numpy()

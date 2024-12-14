# retrieval/multimodal_retrieval.py

from .text_retrieval import TextRetrieval
from .pdf_retrieval import ImageRetrieval
from .audio_retrieval import AudioRetrieval
from .video_retrieval import VideoRetrieval

class MultimodalRetrieval:
    def __init__(self):
        self.text_retrieval = TextRetrieval()
        self.image_retrieval = ImageRetrieval()
        self.audio_retrieval = AudioRetrieval()
        self.video_retrieval = VideoRetrieval()

    def retrieve(self, query: str, file_data: dict) -> list:
        """
        根据文件类型和查询进行多模态检索。

        :param query: 用户查询字符串。
        :param file_data: 包含文件路径和扩展名的信息。
        :return: 检索结果列表。
        """
        results = []

        for file in file_data:
            modality = file.get("modality")
            file_path = file.get("file_path")

            if modality == "text":
                res = self.text_retrieval.retrieval_csv(query, file_path)
            elif modality == "image":
                res = self.image_retrieval.retrieval_image(query, file_path)
            elif modality == "audio":
                res = self.audio_retrieval.retrieval_audio(query, file_path)
            elif modality == "video":
                res = self.video_retrieval.retrieval_video(query, file_path)
            else:
                res = {"error": f"Unsupported modality: {modality}"}

            results.append(res)

        return results

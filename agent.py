from retrieval.text_retrieval import TextRetrieval
from retrieval.image_retrieval import ImageRetrieval
from retrieval.audio_retrieval import AudioRetrieval
from retrieval.video_retrieval import VideoRetrieval
from llm.azure_openai import AzureLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.text_retrieval = TextRetrieval()
        self.image_retrieval = ImageRetrieval()
        self.audio_retrieval = AudioRetrieval()
        self.video_retrieval = VideoRetrieval()
        self.llm = AzureLLM()

    def process_request(self, query, files_data):
        """
        根据文件扩展名选择处理策略，生成嵌入，检索内容，并统一调用 LLM。

        :param query: 用户查询字符串。
        :param files_data: 文件信息列表，每个文件包含 `file_path` 和 `file_extension`。
        :return: LLM 响应或错误信息。
        """
        messages = []

        for file_data in files_data:
            file_extension = file_data["file_extension"]
            file_path = file_data["file_path"]

            logger.info(f"Processing file: {file_path} with extension: {file_extension}")

            try:
                # 根据文件扩展名选择处理逻辑
                if file_extension == ".csv":
                    context = self.text_retrieval.retrieval_csv(query, file_path)
                elif file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
                    context = self.image_retrieval.retrieval_image(query, file_path)
                elif file_extension in [".mp3", ".wav", ".flac"]:
                    context = self.audio_retrieval.retrieval_audio(query, file_path)
                elif file_extension in [".mp4", ".avi", ".mkv"]:
                    context = self.video_retrieval.retrieval_video(query, file_path)
                else:
                    context = {"error": f"Unsupported file type: {file_extension}"}

                # 将检索结果追加到上下文消息中
                if "error" in context:
                    messages.append({"role": "assistant", "content": context["error"]})
                else:
                    messages.append({"role": "assistant", "content": context["retrieved_data"]})

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                messages.append({"role": "assistant", "content": f"Error processing file {file_path}: {e}"})

        # for msg in messages:
        #     print(f"msg['content']: {msg['content']}, type: {type(msg['content'])}")

        # 合并所有内容为单个字符串
        context_str = "\n".join([str(msg["content"]) if isinstance(msg["content"], str) 
                                else "\n".join([str(item) for item in msg["content"]]) 
                                for msg in messages])

        # 调用 LLM 生成响应
        try:
            llm_response = self.llm.generate(query, context_str)  # 使用合并后的字符串作为上下文
            return {"llm_response": llm_response, "messages": messages}
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}

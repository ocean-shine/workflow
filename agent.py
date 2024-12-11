from retrieval.text_retrieval import TextRetrieval
from retrieval.image_retrieval import ImageRetrieval
from retrieval.audio_retrieval import AudioRetrieval
from retrieval.video_retrieval import VideoRetrieval
from llm.azure_openai import AzureLLM
from llm.prompt_builder import build_system_prompt, build_user_prompt
from retrieval.rag import RAG  # 引入 RAG 类
from qdrant_client import QdrantClient  # Qdrant 客户端
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, use_qdrant=True, qdrant_config=None):
        """
        初始化 Agent 类，支持文本、图像、音频、视频的检索，
        以及通过 RAG 模块向量存储和检索。

        :param use_qdrant: 是否启用 Qdrant 存储，默认启用。
        :param qdrant_config: Qdrant 配置信息，只有启用 Qdrant 时需要提供。
        """
        print("Initializing Agent class.")
        
        # 初始化其他检索模块
        self.text_retrieval = TextRetrieval()
        self.image_retrieval = ImageRetrieval()
        self.audio_retrieval = AudioRetrieval()
        self.video_retrieval = VideoRetrieval()

        # 初始化 LLM
        self.llm = AzureLLM()

        # 初始化 Qdrant 客户端
        self.qdrant_client = None
        self.rag = None

        if use_qdrant and qdrant_config:
            # 验证 Qdrant 配置是否有效
            try:
                required_keys = ["host", "port"]  # 根据需要调整配置字段
                for key in required_keys:
                    if key not in qdrant_config:
                        raise ValueError(f"Missing required config key: {key}")
                
                # 初始化 Qdrant 客户端
                self.qdrant_client = QdrantClient(**qdrant_config)
                print("Qdrant client initialized successfully.")
                
                # 初始化 RAG 类（用于向量存储和检索）
                self.rag = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
                print("RAG initialized with Qdrant.")
            except Exception as e:
                print(f"Error initializing Qdrant: {e}")
        else:
            print("RAG is initialized without Qdrant.")

    def detect_embedding_dimension(self, embeddings):
        """
        根据给定的嵌入，检测其维度。
        
        :param embeddings: 检索返回的嵌入
        :return: 嵌入的维度
        """
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings provided or embeddings list is empty.")
            return None
        # 检测嵌入的维度（假设嵌入是一个二维数组）
        dimension = len(embeddings[0]) if isinstance(embeddings[0], list) else len(embeddings)
        print(f"Detected embedding dimension: {dimension}")
        return dimension

    def process_request(self, query: str, files_data: list) -> dict:
        """
        根据文件扩展名选择处理策略，生成嵌入，检索内容，并统一调用 LLM。

        :param query: 用户查询字符串。
        :param files_data: 文件信息列表，每个文件包含 `file_path` 和 `file_extension`。
        :return: LLM 响应或错误信息。
        """
        print(f"Processing request with query: {query}")
        
        messages = []

        for file_data in files_data:
            file_extension = file_data["file_extension"]
            file_path = file_data["file_path"]

            # 记录文件处理过程
            print(f"Processing file: {file_path} with extension: {file_extension}")

            try:
                # 根据文件扩展名选择处理逻辑
                if file_extension == ".csv":
                    context = self.text_retrieval.retrieval_csv(query, file_path)
                    embedding_type = "text"
                elif file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
                    context = self.image_retrieval.retrieval_image(query, file_path)
                    embedding_type = "image"
                elif file_extension in [".mp3", ".wav", ".flac"]:
                    context = self.audio_retrieval.retrieval_audio(query, file_path)
                    embedding_type = "audio"
                elif file_extension in [".mp4", ".avi", ".mkv"]:
                    context = self.video_retrieval.retrieval_video(query, file_path)
                    embedding_type = "video"
                else:
                    context = {"error": f"Unsupported file type: {file_extension}"}
                    embedding_type = None
                    print(f"Unsupported file type: {file_extension}")

                # 将检索结果追加到上下文消息中
                if "error" in context:
                    messages.append({"role": "assistant", "content": str(context["error"])})
                else:
                    messages.append({"role": "assistant", "content": str(context.get("retrieved_data", ""))})

                # 获取嵌入并检测维度
                if "retrieved_data" in context:
                    embeddings = context.get("embeddings", None)
                    if embeddings is not None:
                        dimension = self.detect_embedding_dimension(embeddings)  # 动态检测维度
                        print(f"Detected embedding dimension for {file_path}: {dimension}")

                        if dimension is not None:
                            modality = embedding_type
                            # 添加到 Qdrant 和 RAG
                            if self.rag:
                                self.rag.add_to_index(embeddings, [file_path] * len(embeddings), modality=modality, dimension=dimension)
                        else:
                            print(f"Could not detect embedding dimension for {file_path}")
                    else:
                        print(f"No embeddings found for {file_path}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                messages.append({"role": "assistant", "content": f"Error processing file {file_path}: {e}"})

        # 合并所有内容为单个字符串
        context_str = "\n".join([str(msg["content"]) if isinstance(msg["content"], str) 
                                else "\n".join([str(item) for item in msg["content"]]) 
                                for msg in messages])

        # 打印和记录 messages 内容
        print(f"Messages: {messages}")

        # 打印和记录 context_str 内容
        print(f"Context String: {context_str}")

        # 构建系统和用户提示
        system_prompt = build_system_prompt()
        user_message_content = {
            "type": "text",
            "text": str(context_str)  # 确保将 context_str 转换为字符串
        }

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm.generate({
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_message_content
            })
            print("LLM response generated successfully.")
            return {"llm_response": llm_response, "messages": messages}
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}

# agent.py
        
from retrieval.text_retrieval import TextRetrieval
from retrieval.pdf_retrieval import PDFRetrieval
from retrieval.audio_retrieval import AudioRetrieval
from retrieval.video_retrieval import VideoRetrieval
from llm.azure_openai import LLMVideo, LLMAudio, LLMPDF, LLMText
from llm.prompt_builder import PromptText, PromptPDF, PromptAudio, PromptVideo  # 导入对应的Prompt类
from retrieval.rag import RAG  # 引入 RAG 类
from qdrant_client import QdrantClient  # Qdrant 客户端
import logging
import os

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
        self.text_retrieval = TextRetrieval(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.pdf_retrieval = PDFRetrieval(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.audio_retrieval = AudioRetrieval(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.video_retrieval = VideoRetrieval(use_qdrant=use_qdrant, qdrant_config=qdrant_config)

        # 获取 LLM 配置信息
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_API_BASE")
        api_version = "2024-08-01-preview"

        # 初始化 LLM
        self.llm_text = LLMText("gpt-4o", api_key, api_base, api_version)
        self.llm_pdf = LLMPDF("gpt-4o", api_key, api_base, api_version)
        self.llm_audio = LLMAudio("gpt-4o", api_key, api_base, api_version)
        self.llm_video = LLMVideo("gpt-4o", api_key, api_base, api_version)
        
        # 初始化对应的 Prompt 类
        self.prompt_text = PromptText()
        self.prompt_pdf = PromptPDF()
        self.prompt_audio = PromptAudio()
        self.prompt_video = PromptVideo()

        # Qdrant 初始化移到每个任务中
        self.rag = None
        self.qdrant_client = None

        # Ensure Qdrant is initialized if needed
        if use_qdrant and qdrant_config:
            self.initialize_qdrant(qdrant_config)

    def initialize_qdrant(self, qdrant_config):
        """每次根据任务动态初始化 Qdrant 客户端"""
        try:
            # 验证 Qdrant 配置是否有效
            required_keys = ["host", "port"]
            for key in required_keys:
                if key not in qdrant_config:
                    raise ValueError(f"Missing required config key: {key}")

            # 初始化 Qdrant 客户端
            self.qdrant_client = QdrantClient(**qdrant_config)
            print("Qdrant client initialized successfully.")

            # 初始化 RAG 类（用于向量存储和检索）
            self.rag = RAG(use_qdrant=True, qdrant_config=qdrant_config)
            print("RAG initialized with Qdrant.")
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")

    # ========================= 处理文本文件 ==========================
    def process_text_retrieval(self, query: str, file_path: str) -> dict:
        """
        处理文本文件检索：提取文本嵌入并进行检索。
        """
        print(f"Processing text file for retrieval: {file_path} with query: {query}")
        
        # 使用 TextRetrieval 类来处理文本文件检索
        context = self.text_retrieval.retrieval_csv(query, file_path)
        embedding_type = "text"

        # 获取文本的嵌入
        embeddings = context.get("embeddings", None)
        if embeddings is not None:
            dimension = self.rag.detect_embedding_dimension(embeddings)
            if dimension is not None:
                # 将嵌入添加到 Qdrant 和 RAG
                if self.rag:
                    self.rag.add_to_index(embeddings, [file_path] * len(embeddings), modality=embedding_type, dimension=dimension)
            else:
                print(f"Could not detect embedding dimension for {file_path}")
        else:
            print(f"No embeddings found for text file {file_path}")

        return context

    def generate_text_llm_response(self, context: dict) -> dict:
        """
        生成文本检索结果的 LLM 响应。
        """
        print("Generating LLM response for text retrieval.")

        # 获取文本检索结果
        text_chunks = [context.get("retrieved_data", "")]

        # 合并检索结果为字符串
        context_str = "\n".join(text_chunks)

        # 构建系统和用户提示
        system_prompt = self.prompt_text.build_system_prompt()
        user_message_content = self.prompt_text.build_user_prompt(context_str)

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm_text.generate_text(
                {"content": system_prompt},  # System prompt
                user_message_content         # User prompt
            )
            print("LLM response generated successfully.")
            return {"llm_response": llm_response, "messages": [{"role": "assistant", "content": llm_response}]}
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}

    # ========================= 处理 PDF 文件 ==========================
    def process_pdf_retrieval(self, query: str, file_path: str) -> dict:
        """
        处理 PDF 文件检索：提取 PDF 数据并进行检索。
        """
        print(f"Processing PDF file for retrieval: {file_path} with query: {query}")
        
        # 使用 PDFRetrieval 类来处理 PDF 文件检索
        context = self.pdf_retrieval.retrieval_pdf(query, file_path)
        embedding_type = "text"

        # 获取 PDF 的嵌入
        embeddings = context.get("embeddings", None)
        if embeddings is not None:
            dimension = self.rag.detect_embedding_dimension(embeddings)
            if dimension is not None:
                # 将嵌入添加到 Qdrant 和 RAG
                if self.rag:
                    self.rag.add_to_index(embeddings, [file_path] * len(embeddings), modality=embedding_type, dimension=dimension)
            else:
                print(f"Could not detect embedding dimension for {file_path}")
        else:
            print(f"No embeddings found for PDF file {file_path}")

        return context

    def generate_pdf_llm_response(self, context: dict) -> dict:
        """
        生成 PDF 检索结果的 LLM 响应。
        """
        print("Generating LLM response for PDF retrieval.")

        # 获取 PDF 检索结果
        pdf_text = context.get("retrieved_data", "")

        # 合并检索结果为字符串
        context_str = pdf_text

        # 构建系统和用户提示
        system_prompt = self.prompt_pdf.build_system_prompt()
        user_message_content = self.prompt_pdf.build_user_prompt(context_str)

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm_pdf.generate_pdf(
                {"content": system_prompt},  # System prompt
                user_message_content         # User prompt
            )
            print("LLM response generated successfully.")
            return {"llm_response": llm_response, "messages": [{"role": "assistant", "content": llm_response}]}
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}

    # ========================= 处理音频文件 ==========================
    def process_audio_retrieval(self, query: str, file_path: str) -> dict:
        """
        处理音频文件检索：提取音频数据并进行检索。
        """
        print(f"Processing audio file for retrieval: {file_path} with query: {query}")
        
        # 使用 AudioRetrieval 类来处理音频文件检索
        context = self.audio_retrieval.retrieval_audio(query, file_path)
        embedding_type = "audio"

        # 获取音频的嵌入
        embeddings = context.get("embeddings", None)
        if embeddings is not None:
            dimension = self.rag.detect_embedding_dimension(embeddings)
            if dimension is not None:
                # 将嵌入添加到 Qdrant 和 RAG
                if self.rag:
                    self.rag.add_to_index(embeddings, [file_path] * len(embeddings), modality=embedding_type, dimension=dimension)
            else:
                print(f"Could not detect embedding dimension for {file_path}")
        else:
            print(f"No embeddings found for audio file {file_path}")

        return context

    def generate_audio_llm_response(self, context: dict) -> dict:
        """
        生成音频检索结果的 LLM 响应。
        """
        print("Generating LLM response for audio retrieval.")

        # 获取音频检索结果
        audio_text = context.get("retrieved_data", "")

        # 合并检索结果为字符串
        context_str = audio_text

        # 构建系统和用户提示
        system_prompt = self.prompt_audio.build_system_prompt()
        user_message_content = self.prompt_audio.build_user_prompt(context_str)

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm_audio.generate_audio(
                {"content": system_prompt},  # System prompt
                user_message_content         # User prompt
            )
            print("LLM response generated successfully.")
            return {"llm_response": llm_response, "messages": [{"role": "assistant", "content": llm_response}]}
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}


    # ========================= 处理请求 ==========================

    def process_video_retrieval(self, query: str, file_path: str) -> dict:
        """
        处理视频检索：
        - 提取视频帧并进行检索。
        - 返回检索的结果和相关嵌入。
        """
        print(f"Processing video file for retrieval: {file_path} with query: {query}")
        
        # 使用 VideoRetrieval 类来处理视频并进行检索
        context = self.video_retrieval.process_video(file_path, query)
        # embedding_type = "video"

        # # 假设视频返回图像路径和图像索引
        # image_paths = context.get("image_paths", [])
        # top_k_image_indices = context.get("top_k_images_indices", [])
        # text_chunks = context.get("retrieved_text", [])  # 假设返回的视频检索数据是文本片段

        # # 获取视频的嵌入
        # image_embeddings = context.get("image_embeddings", None)
        # if image_embeddings is not None:
        #     dimension = self.rag.detect_embedding_dimension(image_embeddings)
        #     if dimension is not None:
        #         # 将嵌入添加到 Qdrant 和 RAG
        #         if self.rag:
        #             self.rag.add_to_index(image_embeddings, [file_path] * len(image_embeddings), modality=embedding_type, dimension=dimension)
        #     else:
        #         print(f"Could not detect embedding dimension for {file_path}")
        # else:
        #     print(f"No embeddings found for video file {file_path}")

        # # 生成检索结果返回字典
        # retrieved_texts = text_chunks  # 直接使用 text_chunks 作为检索的文本
        # retrieved_images = context.get("retrieved_images", [])
        # top_k_text_indices = context.get("top_k_text_indices", [])
        # return {
        #     "retrieved_texts": retrieved_texts,
        #     "retrieved_images": retrieved_images,
        #     "image_paths": image_paths,
        #     "top_k_text_indices": top_k_text_indices,
        #     "top_k_image_indices": top_k_image_indices,
        #     "context": context,
        #     "text_chunks": text_chunks
        # }
        return context


    def generate_video_llm_response(self, context: dict) -> dict:
        """
        生成视频检索结果的 LLM 响应：
        - 使用检索得到的图像路径、文本和索引来生成 LLM 响应。
        """
        print("Generating LLM response for video retrieval.")

        # 获取检索结果
        text_chunks = context.get("text_chunks", [])  # 获取检索到的文本
        retrieved_texts = context.get("retrieved_texts", [])  # 获取检索到的文本
        image_paths = context.get("image_paths", [])  # 获取检索到的图像
        top_k_text_indices = context.get("top_k_text_indices", [])  # 获取相关文本的索引
        image_per_text_indices = context.get("image_per_text_indices", [])  # 获取相关图像的索引

        # 确保检索结果存在
        if not retrieved_texts or not text_chunks:
            return {"error": "No retrieval results found."}

        # 处理文本和图像内容
        text_imege_message_content = self.prompt_video.process_images_and_text(
            retrieved_texts, image_paths, top_k_text_indices, image_per_text_indices
        )

        # 将检索到的文本块合并为一个字符串
        # context_str = "\n".join([str(msg) for msg in retrieved_texts])

        # 构建系统和用户提示
        system_prompt = self.prompt_video.build_system_prompt()  # 系统提示
        user_message_content = self.prompt_video.build_user_prompt(text_imege_message_content)  # 用户提示

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm_video.generate_video(
                {"content": system_prompt},  # 系统提示
                user_message_content         # 用户提示
            )
            print("LLM response generated successfully.")

            # 调用方法并返回结果
            # related_images_data = self.get_images_for_llm_response( image_paths, image_per_text_indices)

            # 返回 LLM 响应和相关图像的组合
            return {
                "llm_response": llm_response,
                "messages": context  # 每个文本对应的相关图像
            }
                    
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}

    # def get_images_for_llm_response(self, llm_response, image_paths, image_per_text_indices):
    #     """
    #     根据 LLM 响应选择相关的图像。
    #     :param llm_response: LLM 返回的响应文本。
    #     :param image_paths: 所有提取的视频帧路径。
    #     :param image_per_text_indices: 与文本块最相关的图像索引。
    #     :return: 相关的图像路径列表。
    #     """
    #     # 这里可以实现更复杂的逻辑来映射 LLM 响应中的内容到图像，
    #     # 比如分析 LLM 响应并基于响应文本选择图像。
        
    #     # 假设我们根据 text 索引来返回对应的图像
    #     response_data = []

    #     # 遍历每个文本块及其对应的图像
    #     for idx, text in enumerate(llm_response['texts']):
    #         related_images = []
    #         # 获取该文本块对应的 top_k 图像
    #         for image_idx in image_per_text_indices[idx]:
    #             related_images.append(image_paths[image_idx])
            
    #         # 将该文本块与它的相关图像一起添加到返回结果
    #         response_data.append({
    #             "text": text,
    #             "related_images": related_images
    #         })

    #     return response_data

    # ========================= 处理请求 ==========================
    
    def process_request(self, query: str, files_data: list) -> dict:
        """
        根据文件扩展名选择处理策略，生成嵌入，检索内容，并统一调用 LLM。

        :param query: 用户查询字符串。
        :param files_data: 文件信息列表，每个文件包含 `file_path` 和 `file_extension`。
        :return: LLM 响应或错误信息。
        """
        print(f"Processing request with query: {query}")
        
        messages = []

        # 依次处理每个文件
        for file_data in files_data:
            file_extension = file_data["file_extension"]
            file_path = file_data["file_path"]

            # 记录文件处理过程
            print(f"Processing file: {file_path} with extension: {file_extension}")

            try:
                # 每次处理文件时初始化 Qdrant（如果尚未初始化）
                if self.rag is None:
                    print("Initializing Qdrant for this task...")
                    self.initialize_qdrant(qdrant_config)

                if file_extension == ".csv":
                    context = self.process_text_retrieval(query, file_path)
                    result = self.generate_text_llm_response(context)
                    messages.append(result["messages"][0])  # 返回文本处理的结果
                elif file_extension == ".pdf":
                    context = self.process_pdf_retrieval(query, file_path)
                    result = self.generate_pdf_llm_response(context)
                    messages.append(result["messages"][0])  # 返回 PDF 处理的结果
                elif file_extension in [".mp3", ".wav", ".flac"]:
                    context = self.process_audio_retrieval(query, file_path)
                    result = self.generate_audio_llm_response(context)
                    messages.append(result["messages"][0])  # 返回音频处理的结果
                elif file_extension in [".mp4", ".avi", ".mkv"]:
                    # 先处理视频的检索
                    video_context = self.process_video_retrieval(query, file_path)
                    # 然后生成视频的 LLM 响应
                    result = self.generate_video_llm_response(video_context)
                    messages.extend(result["messages"])  # 返回视频处理的结果
                    return result  # 直接返回视频处理结果
                else:
                    messages.append({"role": "assistant", "content": f"Unsupported file type: {file_extension}"})

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                messages.append({"role": "assistant", "content": f"Error processing file {file_path}: {e}"})

        return {"messages": messages}

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
            return llm_response
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
            return llm_response
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
            return llm_response
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}


    # ========================= 处理请求 ==========================

    def process_video_retrieval(self, query: str, file_paths: list) -> dict:
        """
        处理多个视频文件的检索：
        - 提取每个视频文件的检索结果，并合并到一个复合字典中。
        """
        print(f"Processing video files for retrieval: {file_paths} with query: {query}")
        
        # 初始化一个字典来存储多个视频的检索结果
        combined_context = {
            "text_chunks": [],
            "retrieved_texts": [],
            "image_paths": [],
            "top_k_text_indices": [],
            "image_per_text_indices": []
        }

        for file_path in file_paths:
            # 使用 VideoRetrieval 类来处理每个视频文件
            context = self.video_retrieval.process_video(file_path, query)
            
            # 合并每个视频的检索结果到 combined_context
            combined_context["text_chunks"].extend(context.get("text_chunks", []))
            combined_context["retrieved_texts"].extend(context.get("retrieved_texts", []))
            combined_context["image_paths"].extend(context.get("image_paths", []))
            combined_context["top_k_text_indices"].extend(context.get("top_k_text_indices", []))
            combined_context["image_per_text_indices"].extend(context.get("image_per_text_indices", []))
        
        return combined_context


    def generate_video_llm_response(self, context: dict) -> dict:
        """
        生成多个视频检索结果的 LLM 响应：
        - 使用检索得到的图像路径、文本和索引来生成 LLM 响应。
        """
        print("Generating LLM response for video retrieval.")

        # 从合并后的 context 获取检索结果
        text_chunks = context.get("text_chunks", [])
        retrieved_texts = context.get("retrieved_texts", [])
        image_paths = context.get("image_paths", [])
        top_k_text_indices = context.get("top_k_text_indices", [])
        image_per_text_indices = context.get("image_per_text_indices", [])

        # 确保检索结果存在
        if not retrieved_texts or not text_chunks:
            return {"error": "No retrieval results found."}

        # 处理文本和图像内容
        text_image_message_content = self.prompt_video.process_images_and_text(
            retrieved_texts, image_paths, top_k_text_indices, image_per_text_indices
        )

        # 构建系统和用户提示
        system_prompt = self.prompt_video.build_system_prompt()  # 系统提示
        user_message_content = self.prompt_video.build_user_prompt(text_image_message_content)  # 用户提示

        # 调用 LLM 生成响应
        try:
            print("Calling LLM for response generation.")
            
            llm_response = self.llm_video.generate_video(
                {"content": system_prompt},  # 系统提示
                user_message_content         # 用户提示
            )
            print("LLM response generated successfully.")

            # 返回 LLM 响应和相关图像的组合
            return llm_response
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"error": f"Error during LLM generation: {e}"}


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

        # 初始化 Qdrant（如果尚未初始化）
        if self.rag is None:
            print("Initializing Qdrant for this task...")
            self.initialize_qdrant(qdrant_config)

        # 分类文件
        text_files = []
        audio_files = []
        video_files = []
        unsupported_files = []

        for file_data in files_data:
            file_extension = file_data["file_extension"]
            file_path = file_data["file_path"]
            
            if file_extension in [".csv", ".txt", ".md", ".json", ".latex", ".doc", ".docx", ".xls", ".xlsx", ".html"]:
                text_files.append(file_path)
            elif file_extension in [".mp3", ".wav", ".flac"]:
                audio_files.append(file_path)
            elif file_extension in [".mp4", ".avi", ".mkv"]:
                video_files.append(file_path)
            else:
                unsupported_files.append(file_path)

        # 如果有视频文件，一次性处理所有视频文件
        if video_files:
            print("Processing video files...")
            video_contexts = self.process_video_retrieval(query, video_files)
            video_response = self.generate_video_llm_response(video_contexts)
            result = {"llm_response":video_response,
                      "messages":video_contexts}
            
        # 如果有音频文件，一次性处理所有音频文件
        if audio_files:
            print("Processing audio files...")
            audio_contexts = self.process_audio_retrieval(query, audio_files)
            audio_response = self.generate_audio_llm_response(audio_contexts)
            result = {"llm_response":audio_response,
                      "messages":audio_contexts}
           
        # 如果有文本文件，一次性处理所有文本文件
        if text_files:
            print("Processing text files...")
            text_contexts = self.process_text_retrieval(query, text_files)
            text_response = self.generate_text_llm_response(text_contexts)
            result = {"llm_response":text_response,
                      "messages":text_contexts}

        # 处理不支持的文件类型
        for file_data in unsupported_files:
            file_extension = file_data["file_extension"]
            result = {"llm_response": "assistant", "messages": f"Unsupported file type: {file_extension}"}

        return result
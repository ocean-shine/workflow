# retrieval/audio_retrieval.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models.audio_transcriber import AudioTranscriber
from models.text_encoder import TextEncoder
from retrieval.rag import RAG

class AudioRetrieval:
    def __init__(self, use_qdrant=True, qdrant_config=None, max_tokens=77):
        """
        初始化音频检索类，加载必要的模型和 RAG。
        """
        # 音频转录模型
        self.transcriber = AudioTranscriber()

        # 文本嵌入编码器
        self.text_encoder = TextEncoder()

        # 为文本和音频分别创建不同的 RAG 实例
        self.rag_text = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.rag_audio = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)

    def process_audio(self, audio_path: str, query: str, max_tokens: int = 77, top_k: int = 5):
        """
        处理音频文件并检索与查询最相关的内容。

        :param audio_path: 音频文件路径。
        :param query: 查询文本。
        :param max_tokens: 每个文本块的最大 token 数量。
        :param top_k: 返回的最相似内容的数量。
        :return: 最相关的文本。
        """
        try:
            # Step 1: 转录音频为文本
            transcription = self.transcriber.transcribe(audio_path)
            print(f"Transcription: {transcription}")

            # Step 2: 将转录文本拆分为块
            tokens = transcription.split()
            transcription_chunks = [
                " ".join(tokens[i:i + max_tokens])
                for i in range(0, len(tokens), max_tokens)
            ]
            print(f"Generated {len(transcription_chunks)} transcription chunks.")

            # Step 3: 计算文本块的嵌入
            text_embeddings = self.text_encoder.encode(transcription_chunks)
            text_embeddings = text_embeddings.astype("float32")  # 转为 NumPy 数组
            print(f"Text embeddings shape: {text_embeddings.shape}")

            # 设置文本嵌入的维度并存储到 Qdrant（文本数据存储在 text_collection 中）
            self.rag_text.set_embedding_dimension(modality="text", embeddings=text_embeddings)
            self.rag_text.add_to_index(text_embeddings, transcription_chunks, modality="text")

            # Step 4: 生成查询嵌入
            query_embedding = self.text_encoder.encode([query])[0]

            # Step 5: 从向量数据库中检索与查询最相似的文本
            retrieved_text_chunks = self.rag_text.retrieve(query_embedding, modality="text", top_k=top_k)
            print(f"Retrieved top {top_k} text chunks: {retrieved_text_chunks}")

            return {"retrieved_text": retrieved_text_chunks}

        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {"error": str(e)}

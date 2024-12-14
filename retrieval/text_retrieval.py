import pandas as pd
import numpy as np
from models.text_encoder import TextEncoder
from retrieval.rag import RAG

class TextRetrieval:
    def __init__(self, use_qdrant=True, qdrant_config=None, max_tokens=77):
        """
        初始化 TextRetrieval 类，加载文本编码器和 RAG。

        :param max_tokens: 每个文本块的最大 token 数量。
        """
        self.text_encoder = TextEncoder()
        # 确保初始化 RAG 的实例
        self.rag = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.max_tokens = max_tokens

    def retrieval_csv(self, query, file_path):
        """
        从 CSV 文件中提取文本并检索。

        :param query: 查询文本。
        :param file_path: CSV 文件路径。
        :return: 包含检索结果的字典。
        """
        try:
            # Step 1: 读取 CSV 文件
            df = pd.read_csv(file_path)
            if "notes" not in df.columns:
                return {"error": "The CSV file must contain a 'notes' column."}

            # 提取文本列并分块
            notes = df["notes"].tolist()
            chunks = []
            for note in notes:
                tokens = note.split()
                chunks.extend(
                    " ".join(tokens[i:i + self.max_tokens])
                    for i in range(0, len(tokens), self.max_tokens)
                )
            print(f"Split notes into {len(chunks)} chunks.")

            # Step 2: 生成文本嵌入向量
            embeddings = self.text_encoder.encode(chunks)
            embeddings = embeddings.astype("float32")  # 确保嵌入是 NumPy float32 类型
            print(f"Generated embeddings of shape: {embeddings.shape}")

            # Step 3: 动态设置文本嵌入的维度
            self.rag.set_embedding_dimension(modality="text", embeddings=embeddings)

            # Step 4: 将嵌入向量添加到索引
            self.rag.add_to_index(embeddings, chunks, modality="text")
            print("Embeddings added to RAG index.")

            # Step 5: 生成查询嵌入
            query_embedding = self.text_encoder.encode([query])[0]
            print(f"Query embedding shape: {query_embedding.shape}")

            # Step 6: 从索引中检索最相关的文本
            retrieved_texts = self.rag.retrieve(query_embedding, modality="text", top_k=5)
            print(f"Retrieved {len(retrieved_texts)} texts from RAG.")

            return {"retrieved_data": retrieved_texts}

        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return {"error": f"Error processing CSV file: {e}"}

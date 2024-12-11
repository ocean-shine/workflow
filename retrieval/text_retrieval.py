import pandas as pd
from models.text_encoder import TextEncoder
from retrieval.rag import RAG

class TextRetrieval:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.rag = RAG(use_qdrant=True, qdrant_config={
            'url': 'http://localhost:6333',
            'port': 6333
        })
        # 确保 Qdrant 集合已初始化
         # 添加这一行来初始化 Qdrant 集合

    def retrieval_csv(self, query, file_path):
        """
        从 CSV 文件中提取文本并检索。
        """
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            if "notes" not in df.columns:
                return {"error": "The CSV file must contain a 'notes' column."}

            # 提取文本列
            notes = df["notes"].tolist()

            # 生成文本嵌入向量
            embeddings = self.text_encoder.encode(notes)

            # 将嵌入向量转换为 NumPy 数组（确保格式兼容）
            embeddings = np.array(embeddings)

            # 动态设置文本嵌入的维度
            self.rag.set_embedding_dimension(modality="text", embeddings=embeddings)

            # 添加到索引，指定 modality 为 'text'
            modality = "text"
            self.rag.add_to_index(embeddings, notes, modality)

            # 生成查询嵌入
            query_embedding = self.text_encoder.encode([query])[0]

            # 从索引中检索结果，传递 modality 参数
            results = self.rag.retrieve(query_embedding, modality)
            return {"retrieved_data": results}

        except Exception as e:
            return {"error": f"Error processing CSV file: {e}"}

import pandas as pd
from models.text_encoder import TextEncoder
from retrieval.rag import RAG

class TextRetrieval:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.rag = RAG(dimension=768)  # Example dimension

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

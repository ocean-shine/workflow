from models.image_encoder import ImageEncoder
from retrieval.rag import RAG

class ImageRetrieval:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.rag = RAG(dimension=512)  # Example dimension

    def retrieval_image(self, query, file_path):
        """
        从图像中提取特征并检索。
        """
        try:
            image_embedding = self.image_encoder.encode(file_path)
            self.rag.add_to_index([image_embedding], [file_path])

            query_embedding = self.image_encoder.encode_text(query)
            results = self.rag.retrieve(query_embedding)
            return {"retrieved_data": results}
        except Exception as e:
            return {"error": f"Error processing image file: {e}"}

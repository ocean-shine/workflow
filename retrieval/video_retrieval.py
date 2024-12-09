from models.video_encoder import VideoEncoder
from retrieval.rag import RAG

class VideoRetrieval:
    def __init__(self):
        self.video_encoder = VideoEncoder()
        self.rag = RAG(dimension=1024)  # Example dimension

    def retrieval_video(self, query, file_path):
        """
        从视频中提取特征并检索。
        """
        try:
            video_embedding = self.video_encoder.encode(file_path)
            self.rag.add_to_index([video_embedding], [file_path])

            query_embedding = self.video_encoder.encode_text(query)
            results = self.rag.retrieve(query_embedding)
            return {"retrieved_data": results}
        except Exception as e:
            return {"error": f"Error processing video file: {e}"}

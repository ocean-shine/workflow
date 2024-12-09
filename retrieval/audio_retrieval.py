from models.audio_encoder import AudioEncoder
from retrieval.rag import RAG

class AudioRetrieval:
    def __init__(self):
        self.audio_encoder = AudioEncoder()
        self.rag = RAG(dimension=768)  # Example dimension

    def retrieval_audio(self, query, file_path):
        """
        从音频中提取特征并检索。
        """
        try:
            audio_embedding = self.audio_encoder.encode(file_path)
            self.rag.add_to_index([audio_embedding], [file_path])

            query_embedding = self.audio_encoder.encode_text(query)
            results = self.rag.retrieve(query_embedding)
            return {"retrieved_data": results}
        except Exception as e:
            return {"error": f"Error processing audio file: {e}"}

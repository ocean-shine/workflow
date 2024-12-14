# retrieval/video_retrieval.py

from sklearn.metrics.pairwise import cosine_similarity
from models.video_encoder import VideoEncoder
from retrieval.rag import RAG
from retrieval.rag import RAG

class VideoRetrieval:
    def __init__(self, use_qdrant=True, qdrant_config=None, max_tokens=77):
        """
        初始化 VideoRetrieval 类。

        :param use_qdrant: 是否使用 Qdrant 作为向量存储。
        :param qdrant_config: Qdrant 配置信息，用于初始化 Qdrant 客户端。
        :param max_tokens: 每个文本块的最大 token 数量。
        """
        self.video_encoder = VideoEncoder()
        
        # 确保 Qdrant 客户端初始化
        self.rag_text = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.rag_video = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.use_qdrant = use_qdrant
        self.max_tokens = max_tokens



    def process_video(self, video_path, query, query_top_k=10, text_top_k =1):
        """
        处理视频并计算与查询的相关性。

        :param video_path: 视频文件路径。
        :param query: 用户查询文本。
        :param top_k: 返回的最相关结果数。
        :return: 检索结果。
        """
        try:
            # Step 1: 提取音频并转录
            audio_path = self.video_encoder.extract_audio(video_path)
            transcription = self.video_encoder.transcribe_audio(audio_path)
            print(f"Transcription: {transcription}")

            # Step 2: 将转录文本拆分为块
            tokens = transcription.split()
            transcription_chunks = [
                " ".join(tokens[i:i + self.max_tokens])
                for i in range(0, len(tokens), self.max_tokens)
            ]
            print(f"Generated {len(transcription_chunks)} transcription chunks.")

            # Step 3: 使用 sbert 模型计算文本块的嵌入
            text_embeddings_sbert = self.video_encoder.encode_text_with_sbert(transcription_chunks)
            text_embeddings_clip = self.video_encoder.encode_text_with_clip(transcription_chunks)
            
            # 确保 Qdrant 和 RAG 已初始化
            if self.rag_text and self.rag_text.qdrant_client:
                self.rag_text.set_embedding_dimension(modality="text_sbert", embeddings=text_embeddings_sbert)
                self.rag_text.add_to_index(text_embeddings_sbert, transcription_chunks, modality="text_sbert")
                self.rag_text.set_embedding_dimension(modality="text_clip", embeddings=text_embeddings_clip)
                self.rag_text.add_to_index(text_embeddings_clip, transcription_chunks, modality="text_clip")

            else:
                print("Qdrant or RAG is not initialized. Skipping embedding storage.")

            # Step 4: 提取视频帧并计算图像嵌入
            frame_paths = self.video_encoder.extract_frames(video_path)
            image_embeddings = self.video_encoder.encode_frames_with_clip(frame_paths)
            
            # 同样，确保 Qdrant 和 RAG 已初始化
            if self.rag_video and self.rag_video.qdrant_client:
                self.rag_video.set_embedding_dimension(modality="video", embeddings=image_embeddings)
                self.rag_video.add_to_index(image_embeddings, frame_paths, modality="video")
            else:
                print("Qdrant or RAG is not initialized. Skipping embedding storage.")

            # Step 5: 计算文本和图像的相似性
            similarities = cosine_similarity(text_embeddings_clip, image_embeddings)

            # Step 6: 为每个文本块找到最相似的图像
            text_to_image_map = {}
            image_per_text_indices = []  # 新增：存储与文本块最相似的图像索引
            
            for text_idx, similarity_scores in enumerate(similarities):
                # 获取与当前文本块最相似的前 K 张图像
                top_image_indices = similarity_scores.argsort()[-text_top_k:][::-1]  # 获取 top_k 个最相似的图像索引
                image_per_text_indices.append(top_image_indices)
                
                for idx in top_image_indices:
                    if transcription_chunks[text_idx] not in text_to_image_map:
                        text_to_image_map[transcription_chunks[text_idx]] = []
                    text_to_image_map[transcription_chunks[text_idx]].append(frame_paths[idx])

            # Step 7: 根据查询生成嵌入
            query_embedding = self.video_encoder.encode_text_with_sbert([query])[0]
            text_similarities = cosine_similarity([query_embedding], text_embeddings_sbert)[0]
            top_k_text_indices = text_similarities.argsort()[-query_top_k:][::-1]

            # 检索与查询最相关的文本块
            retrieved_texts = [transcription_chunks[idx] for idx in top_k_text_indices]
            # 检索与查询最相关的images
            # retrieved_images = [ image_per_text_indices[idx] for idx in top_k_text_indices]

            # top_k_image_indices = []
            # for image_indices in retrieved_images:
            #     # 遍历每个子列表，获取对应的图像路径
            #     top_k_image_indices.append(image_indices for idx in top_k_text_indices)



            # 返回包含图像路径、图像索引和文本数据的字典
            return {
                "image_paths": frame_paths,
                "image_per_text_indices": image_per_text_indices,  # 传递与文本相似的图像索引
                "top_k_text_indices": top_k_text_indices,      # 传递与query相似的文本索引
                "retrieved_texts": retrieved_texts,
                "image_embeddings": image_embeddings,
                "text_chunks": transcription_chunks
            }

        except Exception as e:
            print(f"Error processing video file: {e}")
            return {"error": str(e)}

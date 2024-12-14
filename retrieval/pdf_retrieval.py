# retrieval/pdf_retrieval.py

import fitz  # PyMuPDF
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from retrieval.rag import RAG
import shutil

class PDFRetrieval:
    def __init__(self, use_qdrant=True, qdrant_config=None, max_tokens=77):
        """
        初始化 PDF 检索类，用于基于文本块和图像的检索。
        """
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        # 为文本和图像分别初始化 RAG 实例
        self.rag_text = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.rag_image = RAG(use_qdrant=use_qdrant, qdrant_config=qdrant_config)
        self.max_tokens = max_tokens

    def split_pdf(self, pdf_path, output_folder):
        """
        将 PDF 文件拆分为文本块和图像，并映射二者。

        :param pdf_path: PDF 文件路径
        :param output_folder: 用于存储输出的 JSON 和图像
        :return: 文本块和图像的映射字典
        """
        os.makedirs(output_folder, exist_ok=True)
        pdf_document = fitz.open(pdf_path)
        text_chunks = []
        image_mapping = {}  # key: page_number, value: list of image paths

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # 提取文本块并拆分
            text = page.get_text("blocks")
            page_text_chunks = [block[4] for block in text if len(block[4].strip()) > 0]
            for block_text in page_text_chunks:
                tokens = block_text.split()
                chunks = [
                    " ".join(tokens[i:i + self.max_tokens])
                    for i in range(0, len(tokens), self.max_tokens)
                ]
                text_chunks.extend(chunks)

            # 提取图像
            image_mapping[page_num + 1] = []
            for img_index, image in enumerate(page.get_images(full=True)):
                xref = image[0]
                pix = fitz.Pixmap(pdf_document, xref)
                if pix.n < 5:  # 如果是灰度或 RGB
                    image_path = os.path.join(output_folder, f"page_{page_num + 1}_img_{img_index}.png")
                    pix.save(image_path)
                else:  # 转换为 RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_path = os.path.join(output_folder, f"page_{page_num + 1}_img_{img_index}.png")
                    pix.save(image_path)
                image_mapping[page_num + 1].append(image_path)

        pdf_document.close()

        return text_chunks, image_mapping

    def retrieve_from_pdf(self, query, pdf_path, top_k=5):
        """
        检索与查询相似的文本块和图像。

        :param query: 用户查询文本
        :param pdf_path: PDF 文件路径
        :param top_k: 返回的最相关结果数量
        :return: 检索结果字典，包含文本块和图像
        """
        try:
            output_folder = "pdf_output"
            
            # 获取 PDF 所在目录并创建新文件夹用于存储图像
            pdf_dir = os.path.dirname(pdf_path)
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            image_folder = os.path.join(pdf_dir, pdf_filename)  # 以 PDF 文件名命名的文件夹
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            # 1. 将 PDF 分割为文本块和图像
            text_chunks, image_mapping = self.split_pdf(pdf_path, output_folder)
            
            # 更新图像路径，使其存储在新创建的文件夹中
            updated_image_mapping = {}
            for key, image_paths in image_mapping.items():
                updated_image_paths = []
                for img_path in image_paths:
                    # 获取文件名并更新路径
                    img_filename = os.path.basename(img_path)
                    updated_img_path = os.path.join(image_folder, img_filename)
                    shutil.move(img_path, updated_img_path)  # 将图像移动到新的文件夹
                    updated_image_paths.append(updated_img_path)
                updated_image_mapping[key] = updated_image_paths
            image_mapping = updated_image_mapping  # 更新 image_mapping

            # 2. 生成文本块嵌入
            text_embeddings = self.text_encoder.encode(text_chunks)
            text_embeddings = text_embeddings.astype("float32")
            
            # 将文本嵌入存储到 Qdrant，使用专门为文本创建的 RAG 实例
            self.rag_text.set_embedding_dimension(modality="text", embeddings=text_embeddings)
            self.rag_text.add_to_index(text_embeddings, text_chunks, modality="text")

            # 3. 生成图像嵌入
            image_paths = [path for paths in image_mapping.values() for path in paths]
            if image_paths:  # 确保 PDF 中包含图像
                image_embeddings = self.image_encoder.encode_frames(image_paths)
                image_embeddings = image_embeddings.astype("float32")
                
                # 将图像嵌入存储到 Qdrant，使用专门为图像创建的 RAG 实例
                self.rag_image.set_embedding_dimension(modality="pdf", embeddings=image_embeddings)
                self.rag_image.add_to_index(image_embeddings, image_paths, modality="pdf")

                # 计算文本与图像的相似性
                similarities = cosine_similarity(text_embeddings, image_embeddings)

                # 为每个文本块找到最相似的图像
                text_to_image_map = {}
                for text_idx, similarity_scores in enumerate(similarities):
                    top_image_idx = similarity_scores.argsort()[-1]  # 取最相似的图像
                    text_to_image_map[text_chunks[text_idx]] = image_paths[top_image_idx]

            # 4. 对查询进行编码
            query_embedding = self.text_encoder.encode([query])[0]

            # 5. 检索与查询相似的文本块
            text_similarities = cosine_similarity([query_embedding], text_embeddings)[0]
            top_k_text_indices = text_similarities.argsort()[-top_k:][::-1]
            retrieved_texts = [text_chunks[i] for i in top_k_text_indices]

            # 6. 找到这些文本块相关的图像
            retrieved_images = [
                text_to_image_map[text]
                for text in retrieved_texts
                if text in text_to_image_map
            ]

            # 7. 清理生成的文件
            for path in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, path))
            os.rmdir(output_folder)

            return {"retrieved_texts": retrieved_texts, "retrieved_images": retrieved_images}

        except Exception as e:
            print(f"Error retrieving from PDF: {e}")
            return {"error": str(e)}

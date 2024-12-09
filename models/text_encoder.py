from sentence_transformers import SentenceTransformer
import numpy as np
import os





class TextEncoder:
    def __init__(self, model_name="sentence-transformers/paraphrase-distilroberta-base-v1"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        embeddings = self.model.encode(texts)
        return np.array(embeddings).astype('float32')
    
# 取消 HTTP 和 HTTPS 代理
# os.environ.pop('HTTP_PROXY', None)
# os.environ.pop('HTTPS_PROXY', None)


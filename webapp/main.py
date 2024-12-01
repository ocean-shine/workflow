import os
import openai
from openai import AzureOpenAI
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(".env")

# 设置 OpenAI API 基础配置
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_type = "azure"
api_version = "2024-08-01-preview"
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    azure_endpoint = api_base,
    api_version = api_version
)
# 配置 FastAPI
app = FastAPI()

# 初始化 Sentence-Transformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 读取 CSV 文件（假设文件名为 wine-ratings.csv）
wine_data = pd.read_csv('wine-ratings.csv')

# 打印数据检查
print(wine_data.head())

# 获取需要处理的列（假设我们用 "notes" 列作为文本内容）
descriptions = wine_data['notes'].tolist()

# 生成文本嵌入
embeddings = model.encode(descriptions)

# 将嵌入转换为 numpy 数组
embedding_matrix = np.array(embeddings).astype('float32')

# 创建 FAISS 索引
dimension = embedding_matrix.shape[1]  # 嵌入的维度
faiss_index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离度量
faiss_index.add(embedding_matrix)  # 添加所有嵌入到 FAISS 索引

# 定义查询输入类
class Body(BaseModel):
    query: str

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)

@app.post('/ask')
def ask(body: Body):
    """
    使用用户输入的查询，执行 FAISS 搜索并返回最相关的结果。
    """
    search_result = search(body.query)  # 调用 FAISS 搜索
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}

def search(query):
    """
    执行 FAISS 搜索，返回最相关的文档。
    """
    # 为查询生成嵌入
    query_embedding = model.encode([query])[0]
    
    # 使用 FAISS 搜索最相关的嵌入
    query_embedding = np.array([query_embedding]).astype('float32')
    _, indices = faiss_index.search(query_embedding, k=5)  # 返回最相似的前 5 个文档
    
    # 获取最相关的文档内容
    relevant_docs = wine_data.iloc[indices[0]]['notes'].tolist()
    return relevant_docs

def assistant(query, context):
    """
    生成 OpenAI 的对话响应，基于用户的查询和从 FAISS 检索到的上下文。
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that helps people find the best wine based on their preferences."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": " ".join(context)}  # 将检索到的文档内容作为上下文
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # 使用 GPT-4 或你配置的有效模型
        messages=messages
    )

    return response['choices'][0]['message']['content']

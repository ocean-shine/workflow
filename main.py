from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORSMiddleware
from typing import List
import os
import shutil
import base64
import json
import numpy as np
from agent import Agent  # 假设你有一个 Agent 类来处理查询
import logging

# 设置 HTTP 和 HTTPS 代理（如果需要）
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def save_uploaded_files(files: List[UploadFile], data_folder: str):
    os.makedirs(data_folder, exist_ok=True)  # 确保目标文件夹存在
    saved_files = []
    
    for file in files:
        try:
            file_extension = os.path.splitext(file.filename)[1]

            if not file_extension:
                raise ValueError("Uploaded file must have an extension.")

            file_path = os.path.join(data_folder, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append({"file_path": file_path, "file_extension": file_extension.lower()})
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
            raise ValueError(f"Error saving file {file.filename}: {e}")
    return saved_files


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子目录
                else:
                    os.remove(file_path)  # 删除文件
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")



# 初始化 Agent
qdrant_config = {
    "host": "localhost",       # Qdrant 服务地址
    "port": 6333,              # Qdrant 服务端口
    "https": False,            # 是否使用 HTTPS 连接，如果是 HTTPS，则设为 True
    "timeout": 10              # 可选，设置超时时间
}


# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI()

# 创建 Jinja2 模板实例
templates = Jinja2Templates(directory="templates")

# CORS 设置
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的跨域来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有的 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# 提供静态文件支持（可以通过 URL 访问图像）
data_folder = os.path.join(os.getcwd(), "data")
os.makedirs(data_folder, exist_ok=True)

# 提供静态文件支持
html_output_folder = os.path.join(os.getcwd(), "data/html_output")
os.makedirs(html_output_folder, exist_ok=True)

# 配置静态文件的路由
app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "data")), name="static")


# 用于保存 HTML 文件的路径
def get_html_file_path():
    return os.path.join(html_output_folder, "response.html")


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

@app.get('/response', response_class=HTMLResponse)
async def get_response(request: Request):
    """
    处理 GET 请求，显示 HTML 页面。
    """
    query = request.cookies.get("query", "N/A")
    response = request.cookies.get("response", "Waiting for your query")
    messages = request.cookies.get("messages", [])

    logger.info(f"Rendering response page with query: {query}, response: {response}, messages: {messages}")
    print(f"Rendering response page with query: {query}, response: {response}, messages: {messages}")

    # 渲染模板并返回页面
    return templates.TemplateResponse("response.html", {
        "request": request,
        "query": query,
        "response": response,
        "messages": messages
    })


@app.post('/response', response_class=HTMLResponse)
async def ask(query: str = Form(...), files: List[UploadFile] = File(None), request: Request = None):
    """
    处理 POST 请求，接收查询和文件，并返回更新后的 HTML 页面。
    """
    print(f"Received query: {query}")
    print(f"Received {len(files) if files else 0} files.")
    
    try:
        # 删除 data 文件夹中的所有内容
        print(f"Deleting contents of {data_folder} folder.")
        delete_folder_contents(data_folder)

        data = []
        
        # 如果上传了文件，将其保存到 data 文件夹
        if files:
            data = save_uploaded_files(files, data_folder)

        # 调用 Agent 处理
        agent = Agent(use_qdrant=True, qdrant_config=qdrant_config)
        print(f"Processing query with agent.")
        result = agent.process_request(query, data)
        print(f"Agent processed the request successfully.")

        # 获取 LLM 响应和消息
        llm_response = result.get("llm_response", "")
        context = result.get("messages", {})  # 获取所有的消息

        # 提取上下文信息
        image_paths = context.get("image_paths", [])
        image_per_text_indices = context.get("image_per_text_indices", [])
        top_k_text_indices = context.get("top_k_text_indices", [])
        retrieved_texts = context.get("retrieved_texts", [])

        # 图像和文本的处理
        response_data = []
        for text_idx, text_content in zip(top_k_text_indices, retrieved_texts):
            related_images = []
            for image_idx in image_per_text_indices[text_idx]:
                image_path = image_paths[image_idx]
                image_data = f"data:image/jpeg;base64,{image_to_base64(image_path)}"
                related_images.append(image_data)

            # 将文本和相关图像添加到响应数据
            response_data.append({
                "type": "text",
                "text": text_content,
                "images": related_images
            })

        # 渲染模板并保存 HTML 文件
        response_page = templates.TemplateResponse("response.html", {
            "request": request,
            "query": query,
            "response": llm_response,
            "messages": response_data
        })

        # 将生成的 HTML 保存到文件
        html_file_path = get_html_file_path()
        os.makedirs(os.path.dirname(html_file_path), exist_ok=True) 
        with open(html_file_path, "w") as f:
            f.write(response_page.body.decode())  # 将响应页面保存为 HTML 文件

        # 返回文件的 URL（在 static 文件夹下提供访问）
        html_file_url = f"/static/html_output/{os.path.basename(html_file_path)}"
        return RedirectResponse(url=html_file_url, status_code=303)  # 重定向到文件 URL

    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": str(e)}







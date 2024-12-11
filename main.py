from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import RedirectResponse
from agent import Agent
import os
import shutil
import logging
from typing import List

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 HTTP 和 HTTPS 代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 Agent
qdrant_config = {
    "host": "localhost",       # Qdrant 服务地址
    "port": 6333,              # Qdrant 服务端口
    "https": False,            # 是否使用 HTTPS 连接，如果是 HTTPS，则设为 True
    "timeout": 10              # 可选，设置超时时间
}
agent = Agent(use_qdrant=True, qdrant_config=qdrant_config)

# 确保 data 文件夹存在
data_folder = os.path.join(os.getcwd(), "data")
os.makedirs(data_folder, exist_ok=True)


def save_uploaded_files(files: List[UploadFile], data_folder: str):
    """
    保存上传的文件到指定的文件夹，并返回文件路径和扩展名的列表。

    :param files: 上传的文件列表 (FastAPI UploadFile 实例)。
    :param data_folder: 保存文件的目标文件夹。
    :return: 保存的文件路径和文件扩展名的列表。
    """
    os.makedirs(data_folder, exist_ok=True)  # 确保目标文件夹存在
    saved_files = []

    # 打印并记录每个文件的处理状态
    print(f"Saving uploaded files to {data_folder}")
    
    for file in files:
        # 设置断点，查看每个文件的处理过程
        print(f"Processing file: {file.filename}")
        
        try:
            # 提取文件扩展名
            file_extension = os.path.splitext(file.filename)[1]
            print(f"File extension: {file_extension}")

            if not file_extension:
                raise ValueError("Uploaded file must have an extension.")

            # 构造完整的文件路径
            file_path = os.path.join(data_folder, file.filename)
            print(f"Saving file to: {file_path}")

            # 保存文件到目标路径
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 添加保存的文件路径和扩展名到列表
            saved_files.append({"file_path": file_path, "file_extension": file_extension.lower()})
            print(f"File saved: {file.filename}")
            
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
            raise ValueError(f"Error saving file {file.filename}: {e}")

    return saved_files


@app.get('/')
def root():
    """
    重定向到 Swagger 文档页面。
    """
    print("Redirecting to Swagger docs.")
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
async def ask(query: str = Form(...), files: List[UploadFile] = File(None)):
    """
    接收 UI 输入并将请求传递给 Agent 处理。
    支持多文件上传，并将文件保存到当前工作目录的 data 文件夹。

    :param query: 用户查询字符串。
    :param files: 上传的文件列表。
    :return: 处理结果或错误信息。
    """
    # 在接收到请求时输出查询和文件数量
    print(f"Received query: {query}")
    print(f"Received {len(files) if files else 0} files.")

    try:
        data = []
        
        # 如果上传了文件，将其保存到 data 文件夹
        if files:
            print(f"Saving {len(files)} files.")
            # 保存文件并获取文件路径和扩展名
            data = save_uploaded_files(files, data_folder)
        

        # 调用 Agent 处理
        print(f"Processing query with agent.")
        result = agent.process_request(query, data)
        print(f"Agent processed the request successfully.")

        return {"response": result}

    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": str(e)}

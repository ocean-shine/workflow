from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import RedirectResponse
from agent import Agent
import os
import shutil
from typing import List

# 设置 HTTP 和 HTTPS 代理
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 Agent
agent = Agent()

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
    for file in files:
        try:
            # 提取文件扩展名
            file_extension = os.path.splitext(file.filename)[1]
            if not file_extension:
                raise ValueError("Uploaded file must have an extension.")

            # 构造完整的文件路径
            file_path = os.path.join(data_folder, file.filename)

            # 保存文件到目标路径
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 添加保存的文件路径和扩展名到列表
            saved_files.append({"file_path": file_path, "file_extension": file_extension.lower()})
        except Exception as e:
            raise ValueError(f"Error saving file {file.filename}: {e}")

    return saved_files

@app.get('/')
def root():
    """
    重定向到 Swagger 文档页面。
    """
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
    try:
        data = []
        # 如果上传了文件，将其保存到 data 文件夹
        if files:
            data = save_uploaded_files(files, data_folder)

        # 调用 Agent 处理
        result = agent.process_request(query, data)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

import os
import shutil
from typing import List
from agent import Agent
from uuid import uuid4
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 HTTP 和 HTTPS 代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

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


def save_uploaded_files(files: List[str], data_folder: str):
    """
    模拟保存上传的文件到指定的文件夹，并返回文件路径和扩展名的列表。

    :param files: 文件路径列表。
    :param data_folder: 保存文件的目标文件夹。
    :return: 保存的文件路径和文件扩展名的列表。
    """
    os.makedirs(data_folder, exist_ok=True)  # 确保目标文件夹存在
    saved_files = []

    print(f"Saving uploaded files to {data_folder}")
    
    for file_path in files:
        # 打印并记录每个文件的处理状态
        print(f"Processing file: {file_path}")
        
        try:
            # 提取文件扩展名
            file_extension = os.path.splitext(file_path)[1]
            print(f"File extension: {file_extension}")

            if not file_extension:
                raise ValueError("Uploaded file must have an extension.")

            # 构造目标文件的完整路径
            dest_file_path = os.path.join(data_folder, os.path.basename(file_path))
            print(f"Saving file to: {dest_file_path}")

            # 检查源文件路径和目标文件路径是否相同
            if os.path.abspath(file_path) == os.path.abspath(dest_file_path):
                print(f"Source and destination are the same for {file_path}. Skipping copy.")
            else:
                # 如果源路径和目标路径不同，则进行复制
                shutil.copy(file_path, dest_file_path)
                print(f"File saved: {file_path}")

            # 添加保存的文件路径和扩展名到列表
            saved_files.append({"file_path": dest_file_path, "file_extension": file_extension.lower()})

        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            raise ValueError(f"Error saving file {file_path}: {e}")

    return saved_files


def main():
    # 模拟输入数据
    query = "What are the key decision-making models?"
    # 本地文件路径，可以根据需要修改
    files = [
        "/home/ocean/code/workflow/data/decision-making-course.mp4"
    ]
    
    print(f"Received query: {query}")
    print(f"Received {len(files)} files.")
    
    try:
        data = []
        
        # 模拟文件上传
        if files:
            print(f"Saving {len(files)} files.")
            # 保存文件并获取文件路径和扩展名
            data = save_uploaded_files(files, data_folder)
        
        # 调用 Agent 处理
        print(f"Processing query with agent.")
        result = agent.process_request(query, data)
        print(f"Agent processed the request successfully.")
        
        # 输出结果
        print(f"Result: {result}")
        return {"response": result}
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main()

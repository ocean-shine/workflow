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

            # 构造完整的文件路径
            dest_file_path = os.path.join(data_folder, os.path.basename(file_path))
            print(f"Saving file to: {dest_file_path}")

            # # 模拟保存文件
            # shutil.copy(file_path, dest_file_path)

            # 添加保存的文件路径和扩展名到列表
            saved_files.append({"file_path": dest_file_path, "file_extension": file_extension.lower()})
            print(f"File saved: {file_path}")
            
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            raise ValueError(f"Error saving file {file_path}: {e}")

    return saved_files
def delete_folder_contents(folder_path):
    """删除指定文件夹中的所有内容"""
    if os.path.exists(folder_path):
        # 删除文件夹内的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子目录
                else:
                    os.remove(file_path)  # 删除文件
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def copy_folder_contents(src_folder, dest_folder):
    """将src_folder中的所有内容复制到dest_folder"""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # 如果目标文件夹不存在，创建它

    # 遍历src_folder并复制所有内容到dest_folder
    for filename in os.listdir(src_folder):
        src_file_path = os.path.join(src_folder, filename)
        dest_file_path = os.path.join(dest_folder, filename)
        
        try:
            if os.path.isdir(src_file_path):
                shutil.copytree(src_file_path, dest_file_path)  # 复制子目录
            else:
                shutil.copy2(src_file_path, dest_file_path)  # 复制文件，并保留元数据
        except Exception as e:
            print(f"Error copying {src_file_path} to {dest_file_path}: {e}")

def main():
    # 模拟输入数据
    query = "summarize the main points of this file."
    # 本地文件路径，可以根据需要修改
    files = [
        "/home/ocean/code/workflow/data/LLMs_A_Journey_Through_Time_and_Architecture.mp4"
    ]
    
    # 获取当前文件所在目录
    current_dir = os.getcwd()

    # 设置 data 和 temp 文件夹路径
    data_folder = os.path.join(current_dir, "data")
    temp_folder = os.path.join(current_dir, "temp")

    
    print(f"Received query: {query}")
    print(f"Received {len(files)} files.")
    
    try:
        # 第一件事：删除data文件夹中的所有内容
        print(f"Deleting contents of {data_folder} folder.")
        delete_folder_contents(data_folder)
        
        # 第二件事：将temp文件夹中的内容复制到data文件夹
        print(f"Copying contents from {temp_folder} to {data_folder}.")
        copy_folder_contents(temp_folder, data_folder)
        

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
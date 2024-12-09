import os
from dotenv import load_dotenv

load_dotenv()

import os

# 获取当前文件所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 设置 HF_HOME 环境变量到当前目录下的 'cache' 文件夹
HF_HOME = os.path.join(current_directory, 'cache')

# 将 HF_HOME 环境变量添加到环境中
os.environ['HF_HOME'] = HF_HOME

# 输出配置的 HF_HOME 目录路径
print(f"HF_HOME is set to: {HF_HOME}")


class Settings:
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
    MODEL_DEPLOYMENT = "gpt-4o"
    # HF_HOME = os.getenv("TRANSFORMERS_CACHE")

settings = Settings()

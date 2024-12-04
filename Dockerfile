# 基于 Python 3.10 的镜像
FROM python:3.10-slim

# 安装 Azure CLI 和其他工具
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    gnupg && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装通用依赖
RUN pip install --no-cache-dir --upgrade pip openai

# 设置工作目录
WORKDIR /webapp

# 复制并安装 Python 依赖
COPY ./requirements.txt /webapp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 复制项目代码到容器
COPY ./webapp /webapp

# 确保 main.py 文件在 /webapp 目录
RUN ls /webapp  # 检查是否成功复制了 main.py

# 启动 FastAPI 应用
CMD ["main:app", "--host", "0.0.0.0", "--port", "8080"]

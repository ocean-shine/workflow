# 基于 Python 3.10 的镜像
FROM python:3.10-slim

# 安装 Azure CLI 和其他工具
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    vim \
    gnupg && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装通用依赖
RUN pip install --no-cache-dir --upgrade pip 

# 设置工作目录
WORKDIR /workflow

# 将 Dockerfile 所在路径的所有文件复制到 Docker 容器中的 /workflow 文件夹
COPY . /workflow

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade -r /workflow/requirements.txt

# 确保 main.py 文件在 /workflow/webapp 目录
RUN ls /workflow/webapp  # 检查是否成功复制了 main.py

# 进入 /workflow/webapp 目录
WORKDIR /workflow/webapp

# 启动 FastAPI 应用并确保输出日志
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--log-level", "info"]

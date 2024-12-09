# 基于 Ubuntu 22.04 的镜像，包含更多的图形库支持
FROM ubuntu:22.04

# 安装基础依赖和 Python 环境
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    jq \
    vim \
    gnupg \
    libgl1-mesa-glx \    # 安装 libGL.so.1 依赖
    libglib2.0-0 \        # 安装图形库
    libsm6 \              # 安装额外依赖，可能对 OpenCV 有帮助
    libxext6 \            # 图形相关的依赖
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 pip 和其他 Python 依赖
RUN python3 -m pip install --no-cache-dir --upgrade pip 

# 设置工作目录
WORKDIR /workflow

# 将 Dockerfile 所在路径的所有文件复制到 Docker 容器中的 /workflow 文件夹
COPY . /workflow

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade -r /workflow/requirements.txt

# 进入 /workflow/webapp 目录
WORKDIR /workflow/webapp

# 启动 FastAPI 应用并确保输出日志
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--log-level", "info"]

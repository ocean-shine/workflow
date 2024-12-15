# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 更新 apt 包索引并安装系统依赖，包括 Python 和编译工具
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    vim \
    ffmpeg \
    gnupg \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    python3-dev \
    libmupdf-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 删除现有的 pip 符号链接并创建新的符号链接
RUN rm -f /usr/bin/python /usr/bin/pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# 升级 pip 并安装通用依赖
RUN pip install --no-cache-dir --upgrade pip

# 设置工作目录
WORKDIR /workflow

# 将 Dockerfile 所在路径的所有文件复制到容器中的 /workflow 文件夹
COPY . /workflow

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 环境变量设置（连接 Qdrant）
ENV QDRANT_URL=http://qdrant:6333

# 安装 Docker（用于容器内运行 Qdrant 服务）
RUN apt-get update && apt-get install -y docker.io

# 下载并解压 Qdrant
RUN curl -L https://github.com/qdrant/qdrant/releases/download/v1.12.5/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz

# 启动 Qdrant 和 Uvicorn
CMD sh -c "docker run --rm -d -p 6333:6333 qdrant/qdrant:latest && uvicorn main:app --host 0.0.0.0 --port 80"

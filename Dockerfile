# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV QDRANT_URL=http://localhost:6333

# 更新系统和安装依赖
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
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 删除现有的 pip 符号链接并创建新的符号链接
RUN rm -f /usr/bin/python /usr/bin/pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# 升级 pip 并安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip

# 下载并解压 Qdrant 二进制文件
RUN curl -L https://github.com/qdrant/qdrant/releases/download/v1.12.5/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /usr/local/bin

# 设置工作目录
WORKDIR /workflow

# 复制项目文件
COPY . /workflow

# 安装 Python 项目依赖
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Supervisor 配置文件
RUN echo '[supervisord]' > /etc/supervisord.conf && \
    echo 'nodaemon=true' >> /etc/supervisord.conf && \
    echo '[program:qdrant]' >> /etc/supervisord.conf && \
    echo 'command=/usr/local/bin/qdrant' >> /etc/supervisord.conf && \
    echo 'stdout_logfile=/dev/stdout' >> /etc/supervisord.conf && \
    echo 'stderr_logfile=/dev/stderr' >> /etc/supervisord.conf && \
    echo 'stdout_logfile_maxbytes=0' >> /etc/supervisord.conf && \
    echo 'stderr_logfile_maxbytes=0' >> /etc/supervisord.conf && \
    echo '[program:uvicorn]' >> /etc/supervisord.conf && \
    echo 'command=uvicorn main:app --host 0.0.0.0 --port 80 --log-level debug' >> /etc/supervisord.conf && \
    echo 'stdout_logfile=/dev/stdout' >> /etc/supervisord.conf && \
    echo 'stderr_logfile=/dev/stderr' >> /etc/supervisord.conf && \
    echo 'stdout_logfile_maxbytes=0' >> /etc/supervisord.conf && \
    echo 'stderr_logfile_maxbytes=0' >> /etc/supervisord.conf

# 使用 supervisord 同时运行 Qdrant 和 Uvicorn
CMD ["supervisord", "-c", "/etc/supervisord.conf"]


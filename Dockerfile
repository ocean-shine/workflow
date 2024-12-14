FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    vim \
    ffmpeg \
    gnupg \
    libgl1-mesa-glx \
    libsm6 libxext6 libxrender-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  
# 安装 curl, jq, vim, ffmpeg, gnupg, OpenCV, PDF 解析和图像相关依赖，安装 Azure CLI，清理缓存

RUN pip install --no-cache-dir qdrant-client  
# 安装 Qdrant 客户端

RUN pip install --no-cache-dir opencv-python-headless  
# 安装 OpenCV 库

RUN pip install --no-cache-dir PyMuPDF  
# 安装 PyMuPDF 库，用于 PDF 处理

RUN pip install --no-cache-dir --upgrade pip  
# 升级 pip

WORKDIR /workflow  
# 设置工作目录

COPY . /workflow  
# 将代码复制到容器的工作目录

RUN pip install --no-cache-dir --upgrade -r requirements.txt  
# 安装项目依赖

ENV QDRANT_URL=http://qdrant:6333  
# 设置 Qdrant 服务地址

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--log-level", "info"]  
# 启动 FastAPI 应用

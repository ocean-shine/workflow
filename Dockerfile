# 基于 Python 3.10 的镜像
FROM python:3.10-slim

# 安装 OpenCV 依赖的系统库
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    vim \
    gnupg \
    libgl1-mesa-glx \     # 安装 libGL.so.1 依赖
    libglib2.0-0 \        # 安装一些其他图形库的依赖（视情况而定）
    libsm6 \              # 安装额外依赖，可能对 OpenCV 有帮助
    libxext6 \            # 图形相关的依赖
    libxrender1 \         # 提供图形渲染支持
    libfontconfig1 \      # 提供字体配置支持
    libx11-6 \            # 其它 X11 相关依赖
    libv4l-dev \          # 视频设备的开发库
    ffmpeg \              # 安装视频解码支持（对于视频读取很重要）
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 OpenCV
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir opencv-python-headless

# 设置工作目录
WORKDIR /app

# 将代码复制到容器中
COPY . /app

# 安装其他 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

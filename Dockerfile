# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置环境变量，确保交互模式不会影响安装
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    vim \
    gnupg \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libx11-6 \
    libv4l-dev \
    ffmpeg \
    build-essential \
    cmake \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 libGL.so.1 依赖
# 安装额外依赖，可能对 OpenCV 有帮助
# 图形相关的依赖
# 提供图形渲染支持
# 字体配置支持
# X11 相关的依赖
# 视频设备的开发库
# 视频解码支持（对于视频读取很重要）
# 安装编译工具
# 安装 cmake，用于构建 OpenCV
# 安装一些其他图形库的依赖（视情况而定）


# 安装 Python 和 pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev

# 升级 pip
RUN pip3 install --no-cache-dir --upgrade pip

# 安装 OpenCV，使用 headless 版本
RUN pip3 install --no-cache-dir opencv-python-headless

# 设置工作目录
WORKDIR /workflow

# 将本地代码复制到容器中的 /workflow 目录
COPY . /workflow

# 安装 Python 依赖
RUN pip3 install --no-cache-dir -r /workflow/requirements.txt

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

FROM python:3.10

# 设置工作目录
WORKDIR /webapp

# 复制并安装依赖
COPY ./requirements.txt /webapp/requirements.txt
RUN pip install -r requirements.txt

# 复制项目代码
COPY . /webapp

# 启动 FastAPI 应用
ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "80"]

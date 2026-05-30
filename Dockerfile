FROM python:3.11-slim

WORKDIR /app

# libjpeg62-turbo/zlib1g 给 Pillow；其余给 build123d 依赖的 cadquery-ocp
# (OpenCascade 原生库)，headless 也需要 libGL 等，否则 import build123d 会失败。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
        libxi6 \
        libfontconfig1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 验证 build123d 能 import；缺库会让构建直接失败，避免上线后才发现
RUN python -c "import build123d; print('build123d', build123d.__version__)"

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

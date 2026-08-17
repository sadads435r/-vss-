FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# ffmpeg: 抓取 RTSP 当前帧供 ROI 标定（current_picture 不再依赖 VST）
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY workstation.py .
COPY static ./static

USER 65532:65532
ENTRYPOINT ["python", "/app/app.py"]

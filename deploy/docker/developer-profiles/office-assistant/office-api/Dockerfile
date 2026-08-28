FROM python:3.12-slim

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1,10.2.0.0/24"

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY workstation.py .
COPY motion.py .
COPY motion_worker.py .
COPY pose_collector.py .
COPY flywheel.py .
COPY flywheel_worker.py .
COPY gdino_client.py .
COPY nv_pb2.py .
COPY static ./static

USER 65532:65532
ENTRYPOINT ["python", "/app/app.py"]

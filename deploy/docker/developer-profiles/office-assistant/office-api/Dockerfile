FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY workstation.py .
COPY static ./static

USER 65532:65532
ENTRYPOINT ["python", "/app/app.py"]

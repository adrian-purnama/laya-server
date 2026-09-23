FROM python:3.12-slim
 
WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY server.py .
 

ENV HOST=0.0.0.0
ENV PORT=8000
ENV HF_TOKEN=""

 
EXPOSE 8000
 
VOLUME ["/root/.cache/huggingface"]
 
CMD ["python", "server.py"]
 
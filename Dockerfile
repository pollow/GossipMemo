FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOSSIPMEMO_DATABASE_PATH=/data/gossipmemo.db

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[server]"

VOLUME ["/data"]
EXPOSE 8765

CMD ["gossipmemo", "serve", "--host", "0.0.0.0", "--port", "8765"]

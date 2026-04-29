FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U "opendataloader-pdf[hybrid]" fastapi uvicorn

ENV JAVA_TOOL_OPTIONS=-Djava.util.logging.ConsoleHandler.level=WARNING

WORKDIR /app
COPY server.py .

EXPOSE 5002

CMD ["sh", "-c", "opendataloader-pdf-hybrid --port 5010 --device cpu & uvicorn server:app --host 0.0.0.0 --port 5002"]

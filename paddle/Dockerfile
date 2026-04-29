FROM paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed \
        "paddleocr==3.4.0" \
        "paddlex[ocr]==3.4.3" \
        fastapi \
        uvicorn

WORKDIR /app
COPY server.py .

EXPOSE 5003

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5003"]
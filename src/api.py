"""FastAPI 진입점.

uvicorn이 로드하는 ASGI 앱. v1 router를 /api/v1/docs-rag prefix로 마운트.
celery_app을 import하는 이유: API 프로세스에서도 broker 연결을 살려둬야
worker 죽었을 때 task 발행이 큐에 쌓임 (앱 자체가 죽지 않음).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from celery_app import celery_app  # noqa: F401 — API 프로세스에서 broker 연결 보장
from v1.router import router as v1_router

app = FastAPI(
    title="Docs RAG API",
    description="문서 RAG API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1/docs-rag")

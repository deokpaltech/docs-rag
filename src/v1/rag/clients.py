"""공유 자원 싱글톤 — Qdrant client / LLM / CrossEncoder reranker.

Module import 시점에 1회 초기화. CrossEncoder 모델 로드가 무거우므로 (~1GB)
프로세스당 한 번만. router·search 등 여러 모듈이 import해도 동일 인스턴스 공유.

테스트에서 mock 필요 시 `patch("src.v1.rag.clients.llm")` 등으로 단일 진입점 mocking.
"""
from __future__ import annotations

import re

from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from ..config import LLM_CONFIG, QDRANT_CONFIG, RERANKER_CONFIG

qdrant = QdrantClient(
    host=QDRANT_CONFIG["host"],
    port=QDRANT_CONFIG["port"],
    grpc_port=QDRANT_CONFIG["grpc_port"],
    prefer_grpc=True,
)

llm = ChatOpenAI(
    model=LLM_CONFIG["model"],
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
    temperature=LLM_CONFIG["options"]["temperature"],
    top_p=LLM_CONFIG["options"]["top_p"],
    max_tokens=LLM_CONFIG["options"]["max_tokens"],
    seed=LLM_CONFIG["options"]["seed"],
)

reranker = CrossEncoder(RERANKER_CONFIG["model"], local_files_only=True)

# Qwen3 등 reasoning 모델이 출력하는 내부 사고 과정 제거.
# 호출처 drift 방지 위해 invoke_clean() 으로 묶음 — 모든 LLM 호출은 이 함수 경유 권장.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def invoke_clean(messages) -> str:
    """LLM 호출 + reasoning think 태그 제거 단일 진입점.

    Qwen3·DeepSeek R1 같은 reasoning 모델은 답변 앞에 `<think>...</think>` 로
    내부 사고 과정을 같이 출력함. 사용자한테 노출되면 안 되므로 strip.
    이 함수를 모든 LLM 호출의 단일 진입점으로 두면 drift 방지.
    """
    return THINK_RE.sub("", llm.invoke(messages).content).strip()

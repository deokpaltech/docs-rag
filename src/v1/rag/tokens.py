"""LLM context 토큰 예산 계산 + 섹션 단위 트렁케이션.

목적: vLLM max_context(8192) 초과 시 silent truncate로 답변 끝/시스템 프롬프트가
잘려서 품질 저하. 사전 계산해 안전 한도(예산) 안에서 컨텍스트를 자른다.

자르는 방식 — greedy 섹션 추가:
  reranking 순서대로 섹션을 보고, 들어가면 통째로 추가 / 안 들어가면 통째로
  스킵하고 다음 섹션 시도. 섹션 중간을 자르지 않아 헤딩·표·조항 인용이 깨지지
  않음. DP knapsack 안 쓰는 이유 — 한국어 약관 5~15 섹션 규모라 이득 미미.

Qwen3 tokenizer를 vLLM 서빙 모델과 동일하게 사용 (다른 토크나이저면 budget
계산이 어긋남). AutoTokenizer 로드 실패 시 tiktoken 폴백 (transformers 5.x의
Qwen2Tokenizer 호환 문제 우회).
"""
from __future__ import annotations

import os

from transformers import AutoTokenizer

from ..config import LLM_CONFIG

try:
    _tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("LLM_TOKENIZER_PATH", "/app/model/Qwen3-14B-AWQ"),
        local_files_only=True,
    )
except Exception:
    _tokenizer = None


def count_tokens(text: str) -> int:
    """Qwen3 tokenizer로 토큰 수 계산. AutoTokenizer 실패 시 tiktoken 폴백."""
    if _tokenizer is not None:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def calc_context_budget(system_prompt: str, user_query: str) -> int:
    """프롬프트/출력/여유분을 빼고 context에 쓸 수 있는 토큰 수."""
    max_context = LLM_CONFIG["max_context"]
    max_tokens = LLM_CONFIG["options"]["max_tokens"]
    safety = LLM_CONFIG["safety_margin"]
    prompt_tokens = count_tokens(system_prompt) + count_tokens(user_query)
    return max_context - max_tokens - prompt_tokens - safety


def truncate_context(context: str, budget: int) -> str:
    """예산 내에서 섹션('---' 구분) 단위 greedy knapsack.
    reranking 순서(관련도 높은 순) 유지하면서 예산 초과 큰 섹션은 스킵 + 뒤쪽 작은 섹션 채움.
    전체를 한 번 토큰화 후 섹션별 글자 수 비율로 토큰 수 역산 (N번 → 1번 최적화).
    """
    sections = context.split("\n\n---\n\n")
    separator = "\n\n---\n\n"

    all_tokens = _tokenizer.encode(context, add_special_tokens=False) if _tokenizer else context.encode()
    total_tokens = len(all_tokens) if _tokenizer else len(context) // 2  # 폴백: 1글자 ≈ 0.5토큰

    if total_tokens <= budget:
        return context

    total_chars = len(context)
    tokens_per_char = total_tokens / total_chars if total_chars > 0 else 1
    section_tokens = [max(1, int(len(s) * tokens_per_char)) for s in sections]

    selected_indices = []
    used = 0
    for i, tokens in enumerate(section_tokens):
        if used + tokens <= budget:
            selected_indices.append(i)
            used += tokens

    if not selected_indices:
        tokens = all_tokens[:budget] if _tokenizer else []
        return _tokenizer.decode(tokens) if _tokenizer else context[:budget * 2]

    return separator.join(sections[i] for i in selected_indices)

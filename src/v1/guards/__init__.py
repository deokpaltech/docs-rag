"""Guardrail 패키지 — 입력/출력/접근/액션 등 안전 계층.

OWASP LLM Top 10의 LLM06 (Sensitive Information Disclosure) · LLM01 (Prompt Injection)
2종에 대한 1단계 정규식 방어 구현. 나머지 4계층(Access · Retrieval · Grounding · Output ·
Action)은 단일 도메인 read-only RAG라 우선순위를 낮춰 deferred — 도메인·인증·tool calling
추가 시 같은 패키지에 모듈 추가.

설계 원칙:
  - 각 guard는 pure function 형태 (state 없음, 테스트 용이)
  - 정규식·heuristic 1단계 → 향후 Microsoft Presidio·AWS Comprehend·GCP DLP (PII)
    Lakera Guard·Rebuff·NeMo Guardrails (injection) 같은 LLM judge로 교체 가능하게
    인터페이스 분리.
"""

from .pii import mask_pii, mask_pii_list, PIIKind
from .injection import sanitize_input
from .output import sanitize_output

__all__ = ["mask_pii", "mask_pii_list", "PIIKind", "sanitize_input", "sanitize_output"]

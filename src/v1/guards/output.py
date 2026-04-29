"""Output Guard — LLM 답변 후처리 정제.

OWASP LLM02 (Insecure Output Handling) 1단계 정규식 방어. role token leak·
system prompt 노출·욕설 차단. Anthropic Moderations · OpenAI Moderations ·
Lakera Output Guard 같은 외부 judge로 교체 가능한 인터페이스.

정책:
  - leak 토큰(role marker / ChatML / Llama Instruct): 즉시 silent 제거
  - 욕설: 라벨만 기록 (사용자 응답 단절보다 추적 우선, PII와 같은 패턴)

한계:
  - 정규식 화이트리스트라 변형(난독화·우회) 놓칠 수 있음
  - 도메인별 욕설 사전은 한국어 보험 서비스 기준 최소 — 다른 도메인 적용 시 확장
  - 의미 기반 toxicity / 편향 미커버 → LLM moderations API 어댑터 슬롯
"""
import re

# 답변에 LLM의 role token이나 시스템 프롬프트 흔적이 새어 나올 때
_LEAK_PATTERNS = [
    re.compile(r"<\|im_(start|end)\|>"),       # ChatML
    re.compile(r"\[INST\]|\[/INST\]"),           # Llama Instruct
    re.compile(r"<\|endoftext\|>"),              # GPT special
    re.compile(r"^\s*(system|assistant)\s*:\s*", re.MULTILINE | re.IGNORECASE),
]

# 한국어 보험 도메인 기준 최소 욕설 — 운영 데이터로 검증된 후 확장
_PROFANITY_PATTERNS = [
    re.compile(r"(시발|씨발|개새끼|병신|좆|fuck|shit)", re.IGNORECASE),
]


def sanitize_output(answer: str) -> tuple[str, list[str]]:
    """답변 정제 + 발견된 위협 라벨 반환.

    leak 토큰은 silent 제거 (사용자에게 system 흔적 노출 방지).
    욕설은 정규식 매칭만 기록 — 차단·응답 차단은 별도 정책 결정 필요.
    """
    threats: list[str] = []
    for pat in _LEAK_PATTERNS:
        if pat.search(answer):
            threats.append(f"output_leak:{pat.pattern[:30]}")
            answer = pat.sub("", answer)
    for pat in _PROFANITY_PATTERNS:
        if pat.search(answer):
            threats.append(f"profanity:{pat.pattern[:30]}")
    return answer, threats

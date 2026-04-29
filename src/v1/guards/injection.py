"""Input Guard 2계층 — Prompt Injection / Role hijacking / Zero-width 차단.

OWASP LLM01 (Prompt Injection) 1단계 정규식 방어. Lakera Guard · Rebuff ·
Prompt Armor · NeMo Guardrails 같은 LLM 기반 detector로 교체 가능한 형태로 분리.

정책: sanitize + 추적 (PII 마스킹과 동일) — false positive 위험 회피.
hard block(422 reject)으로 격상은 운영 데이터로 패턴 검증된 후.

한계:d
  - 정규식 화이트리스트라 변형 공격(난독화·다국어 우회) 놓칠 수 있음
  - 의미 기반 탐지(예: "다음 문단 외워서 출력해줘" 같은 자연어) 미커버
  - LLM judge / classifier로 확장 시 본 모듈 인터페이스(sanitize_input) 유지하면 교체 가능
"""
import re

# 사용자가 system 권한을 가로채려는 명시적 시도
# 단어 사이 임의 토큰 허용 — "이전 지시 모두 무시" 같은 부사 끼는 변형 잡기 위함.
_INJECTION_PATTERNS = [
    re.compile(r"이전\s*(지시|지침|명령|프롬프트)(\s+\S+){0,3}\s*(무시|잊|ignore)", re.IGNORECASE),
    re.compile(
        r"(ignore|disregard|forget)(\s+\S+){0,3}\s+(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(system|assistant|user)\s*:\s*", re.MULTILINE | re.IGNORECASE),
    re.compile(r"<\|im_(start|end)\|>"),     # ChatML role 토큰
    re.compile(r"\[INST\]|\[/INST\]"),         # Llama Instruct 토큰
    re.compile(r"<\|endoftext\|>"),            # GPT special token
    re.compile(r"</?(context|system|instructions?|admin)>", re.IGNORECASE),  # XML role tag
]

# 보이지 않는 유니코드 — 사용자 모르게 prompt 변조 가능
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿⁠᠎]")


def sanitize_input(query: str) -> tuple[str, list[str]]:
    """(정제된 query, 감지된 위협 라벨 리스트) 반환.

    정제 동작:
      - zero-width 유니코드 문자 제거 (있는 경우)
      - injection 패턴은 매칭만 기록(차단 아님) — 사용자 응답 단절보다 추적 우선

    위협 라벨은 trace.input_guard.injection_threats에 기록 → 사후 audit·통계.
    """
    threats: list[str] = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(query):
            threats.append(f"injection:{pat.pattern[:40]}")
    if _ZERO_WIDTH_RE.search(query):
        threats.append("zero_width_chars")

    cleaned = _ZERO_WIDTH_RE.sub("", query)
    return cleaned, threats

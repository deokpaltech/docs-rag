"""Input Guard — PII 정규식 마스킹 (1단계 방어).

목적: 사용자 쿼리에 포함된 PII가 trace JSONL · DB · LLM 호출에 그대로 흘러들지 않게 차단.
OWASP LLM06 (Sensitive Information Disclosure) 1단계 대응. Microsoft Presidio · AWS
Comprehend · GCP DLP 같은 DLP 도구로 교체 가능한 형태로 분리.

한계 (정직):
  - 정규식 화이트리스트 — 비표준 표기(공백·특수문자 변형)는 놓칠 수 있음
  - 비정형 PII(이름·주소) 미커버 — NER 모델 도입 시 확장
  - 한국 PII 패턴 우선 (주민번호·휴대폰·계좌·카드·이메일)

운영 정책:
  - 마스킹된 텍스트는 LLM·trace에 모두 동일하게 사용 (서빙 결과는 영향받지만 PII 누출이
    훨씬 큰 위험이라 trade-off 수용).
  - 발견된 PII 유형은 trace.input_guard에 메타데이터로 기록 → 사후 audit·통계.

확장 예정:
  - Presidio NER 어댑터 (이름·주소·기관명 같은 비정형 PII)
  - 도메인별 패턴 (의료: 진단코드, 금융: 거래ID 등)
  - Prompt injection 패턴 감지 (Rebuff·Lakera 어댑터)
"""

from __future__ import annotations

import re
from typing import Literal

PIIKind = Literal["RRN", "CARD", "PHONE", "ACCOUNT", "EMAIL"]


# 한국 PII 패턴.
# 주의: 정규식이 너무 느슨하면 false positive (예: 일반 숫자 13자리를 카드로 매칭)가 폭발하므로
# 표준 표기 + 흔한 변형(하이픈/공백) 정도만 커버. 비표준은 NER 단계로 위임.
_PII_PATTERNS: dict[PIIKind, re.Pattern[str]] = {
    # 주민등록번호: YYMMDD-NNNNNNN (하이픈/공백 허용). 외국인등록번호 동일 구조.
    "RRN": re.compile(r"\b\d{6}[-\s]?[1-8]\d{6}\b"),
    # 신용카드: 13~19자리. Luhn 미검증 (false positive 일부 수용 — 차라리 마스킹 과다가 안전).
    "CARD": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,7}\b"),
    # 휴대폰: 010/011/016/017/018/019. 하이픈/공백 허용.
    "PHONE": re.compile(r"\b01[016-9][-\s]?\d{3,4}[-\s]?\d{4}\b"),
    # 은행 계좌: 3-2~6-6+ 형태. CARD 패턴과 충돌 가능 → 순서로 처리 (CARD 먼저).
    "ACCOUNT": re.compile(r"\b\d{3}[-\s]\d{2,6}[-\s]\d{6,}\b"),
    # 이메일.
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

# 정규식 적용 순서 — 더 긴 패턴(CARD, ACCOUNT)을 먼저 적용해서 짧은 패턴(PHONE)에 잠식되는 것 방지.
_APPLY_ORDER: list[PIIKind] = ["RRN", "CARD", "ACCOUNT", "PHONE", "EMAIL"]


def mask_pii(text: str | None) -> tuple[str, list[PIIKind]]:
    """텍스트에서 PII를 감지·마스킹. 발견된 PII 유형 목록도 반환.

    Args:
        text: 사용자 입력 또는 임의 텍스트. None / 빈 문자열은 그대로 반환.

    Returns:
        (masked_text, found_kinds) — found_kinds는 중복 제거된 정렬 목록.

    Examples:
        >>> mask_pii("주민번호 901234-1234567 입니다")
        ('주민번호 [RRN_MASKED] 입니다', ['RRN'])
        >>> mask_pii("연락처: 010-1234-5678, 이메일 a@b.com")
        ('연락처: [PHONE_MASKED], 이메일 [EMAIL_MASKED]', ['EMAIL', 'PHONE'])
        >>> mask_pii("일반 질문입니다")
        ('일반 질문입니다', [])
        >>> mask_pii(None)
        ('', [])
    """
    if not text:
        return "", []

    found: set[PIIKind] = set()
    masked = text
    for kind in _APPLY_ORDER:
        pattern = _PII_PATTERNS[kind]
        if pattern.search(masked):
            found.add(kind)
            masked = pattern.sub(f"[{kind}_MASKED]", masked)

    return masked, sorted(found)


def mask_pii_list(items: list[str] | None) -> tuple[list[str], list[PIIKind]]:
    """리스트(예: include_keywords)의 각 원소에 PII 마스킹 적용.

    빈 리스트/None은 그대로 반환. 발견된 PII 유형은 모든 원소에서 union.
    """
    if not items:
        return items or [], []

    masked_items: list[str] = []
    all_found: set[PIIKind] = set()
    for item in items:
        m, kinds = mask_pii(item)
        masked_items.append(m)
        all_found.update(kinds)
    return masked_items, sorted(all_found)

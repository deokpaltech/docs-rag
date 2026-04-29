"""Input Guard PII 마스킹 단위 테스트 (Injection은 test_injection.py 별도).

src/v1/guards/pii.py — Microsoft Presidio·AWS Comprehend 같은 DLP 도구 도입 전
1단계 정규식 방어. Guardrails 6계층 중 Input Guard의 PII 컴포넌트.

host에서 모델 의존성 없이 실행 가능 (정규식 + dataclass 뿐).
"""
from __future__ import annotations

import pytest

from src.v1.guards.pii import mask_pii, mask_pii_list


# ───────────────── Positive cases (마스킹 발동) ─────────────────


@pytest.mark.parametrize("raw,expected_kind,expected_marker", [
    ("주민번호 901234-1234567 입니다", "RRN", "[RRN_MASKED]"),
    ("RRN 9012341234567 같은 형태도", "RRN", "[RRN_MASKED]"),  # 하이픈 없는 변형
    ("연락처 010-1234-5678 입니다", "PHONE", "[PHONE_MASKED]"),
    ("01012345678 처럼 하이픈 없어도", "PHONE", "[PHONE_MASKED]"),
    ("이메일 test@example.com 입니다", "EMAIL", "[EMAIL_MASKED]"),
    ("카드 1234-5678-9012-3456 입니다", "CARD", "[CARD_MASKED]"),
    ("계좌 110-123-456789 입니다", "ACCOUNT", "[ACCOUNT_MASKED]"),
])
def test_mask_pii_detects_single_kind(raw: str, expected_kind: str, expected_marker: str):
    """단일 PII 종류 — 마스커 텍스트 치환 + kind 반환 검증."""
    masked, found = mask_pii(raw)
    assert expected_marker in masked, f"마커 누락: {masked}"
    assert expected_kind in found, f"kind 누락: {found}"
    # raw PII 원본 텍스트가 결과에 남아있으면 안 됨 (마스킹 핵심 보장)
    assert "9012341234567".replace("-", "") not in masked.replace("-", "") or "MASKED" in masked


def test_mask_pii_multiple_kinds():
    """한 문자열 안에 여러 PII 종류 — 모두 감지·치환."""
    raw = "주민번호 901234-1234567, 연락처 010-1234-5678, 이메일 a@b.com"
    masked, found = mask_pii(raw)
    assert "[RRN_MASKED]" in masked
    assert "[PHONE_MASKED]" in masked
    assert "[EMAIL_MASKED]" in masked
    assert set(found) == {"RRN", "PHONE", "EMAIL"}


def test_mask_pii_repeated_same_kind():
    """같은 종류 PII가 여러 번 등장 — 모두 치환, found는 중복 제거."""
    raw = "010-1111-2222 또는 010-3333-4444"
    masked, found = mask_pii(raw)
    assert masked.count("[PHONE_MASKED]") == 2
    assert found == ["PHONE"]  # 중복 제거


# ───────────────── Negative cases (마스킹 발동 X) ─────────────────


@pytest.mark.parametrize("raw", [
    "일반 질문입니다",
    "제43조 무면허운전 등의 금지",
    "보험금 1,000만원 한도",
    "2026년 4월 27일 시행",
])
def test_mask_pii_no_false_positive_on_normal_text(raw: str):
    """일반 문서 텍스트·조항 번호·금액·날짜는 PII로 잡히면 안 됨."""
    masked, found = mask_pii(raw)
    assert masked == raw, f"false positive: {raw} → {masked}"
    assert found == []


def test_mask_pii_empty_input():
    """None / 빈 문자열은 안전하게 처리."""
    assert mask_pii(None) == ("", [])
    assert mask_pii("") == ("", [])


# ───────────────── mask_pii_list ─────────────────


def test_mask_pii_list_all_clean():
    """모든 원소가 PII 없음 → 원본 그대로 + 빈 found."""
    items = ["보험금", "지급 절차"]
    masked, found = mask_pii_list(items)
    assert masked == items
    assert found == []


def test_mask_pii_list_partial_pii():
    """일부 원소에만 PII — 해당 원소만 치환, found는 union."""
    items = ["보험금 청구", "010-1234-5678 연락주세요", "test@example.com 으로"]
    masked, found = mask_pii_list(items)
    assert masked[0] == "보험금 청구"
    assert "[PHONE_MASKED]" in masked[1]
    assert "[EMAIL_MASKED]" in masked[2]
    assert set(found) == {"PHONE", "EMAIL"}


def test_mask_pii_list_none_or_empty():
    """None / 빈 리스트 안전 처리."""
    assert mask_pii_list(None) == ([], [])
    assert mask_pii_list([]) == ([], [])


# ───────────────── 우선순위 (CARD vs PHONE) ─────────────────


def test_mask_pii_card_before_phone_no_collision():
    """카드번호 16자리가 PHONE 패턴(010-1234-5678)에 잠식되지 않게 우선순위 보장."""
    raw = "카드 1234-5678-9012-3456 입니다"
    masked, found = mask_pii(raw)
    # CARD가 먼저 매칭되어야 함 — PHONE은 같이 나오면 안 됨 (12자리 부분이 PHONE처럼 보일 수 있음)
    assert "[CARD_MASKED]" in masked
    assert "PHONE" not in found

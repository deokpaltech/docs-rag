"""TraceRecord 필드 검증 — critic / input_guard 필드 발행 확인.

schema_version 필드는 의도적으로 두지 않음 (단일 producer + 2 consumer라 schema 협상
비용 불필요, 신규 필드는 optional 추가 + consumer는 .get() 방어적 읽기).
"""
from __future__ import annotations


def test_tracerecord_has_critic_field_default_none():
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    assert hasattr(rec, "critic")
    assert rec.critic is None


def test_tracerecord_critic_accepts_dict():
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.critic = {
        "invoked": True,
        "failure_type": "generation_error",
        "action_taken": "regenerate",
        "before_risk": "hard_fail",
        "after_risk": "pass",
        "regenerate_improved": True,
    }
    assert rec.critic["failure_type"] == "generation_error"
    assert rec.critic["regenerate_improved"] is True


def test_tracerecord_critic_invoked_false():
    """critic이 개입 안 한 케이스 — invoked=False만 기록."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.critic = {"invoked": False}
    assert rec.critic["invoked"] is False


def test_tracerecord_serializes_with_critic():
    """write_trace의 직렬화 경로(asdict + json.dumps)가 critic 필드를 포함해야."""
    import json
    from dataclasses import asdict
    from src.v1.rag.trace import TraceRecord

    rec = TraceRecord()
    rec.critic = {"invoked": True, "failure_type": "unit_error"}
    serialized = json.dumps(asdict(rec), default=str)
    assert '"critic"' in serialized
    assert '"unit_error"' in serialized


def test_tracerecord_default_serialization_critic_is_null():
    """critic 미기록 시 직렬화 결과가 null로 나와야 (backward compat)."""
    import json
    from dataclasses import asdict
    from src.v1.rag.trace import TraceRecord

    rec = TraceRecord()
    serialized = json.loads(json.dumps(asdict(rec), default=str))
    assert serialized["critic"] is None


def test_tracerecord_has_input_guard_field_default_none():
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    assert hasattr(rec, "input_guard")
    assert rec.input_guard is None


def test_tracerecord_input_guard_accepts_dict():
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.input_guard = {"pii_found": ["RRN", "PHONE"], "pii_count": 2}
    assert rec.input_guard["pii_count"] == 2
    assert "RRN" in rec.input_guard["pii_found"]


def test_tracerecord_input_guard_accepts_injection_threats():
    """guards.injection.sanitize_input 결과 — 신규 필드 회귀 방지."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.input_guard = {
        "pii_found": [],
        "pii_count": 0,
        "injection_threats": ["zero_width_chars", "injection:이전\\s*(지시"],
    }
    assert "zero_width_chars" in rec.input_guard["injection_threats"]


def test_tracerecord_verification_accepts_groundedness():
    """groundedness = supported / verifiable — RAGAS faithfulness 패턴.
    verifiable_claims_count == 0 일 땐 groundedness 키 자체 생략 (no signal)."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.verification = {
        "risk_level": "warn",
        "claims_count": 4,
        "verifiable_claims_count": 4,
        "supported_claims_count": 3,
        "groundedness": 0.75,
        "missing_refs_count": 0,
        "numeric_mismatch_count": 1,
    }
    assert rec.verification["groundedness"] == 0.75
    assert 0.0 <= rec.verification["groundedness"] <= 1.0
    assert rec.verification["verifiable_claims_count"] == 4


def test_tracerecord_verification_omits_groundedness_when_no_verifiable_claims():
    """절차형 답변(평문 only) — 모든 claim에 ref/numeric 없음 → groundedness 측정 불가."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.verification = {
        "risk_level": "pass",
        "claims_count": 5,
        "verifiable_claims_count": 0,
        "supported_claims_count": 0,
        "missing_refs_count": 0,
        "numeric_mismatch_count": 0,
    }
    # groundedness 키 없음 → aggregator가 평균에서 자연스럽게 제외
    assert "groundedness" not in rec.verification
    assert rec.verification["verifiable_claims_count"] == 0


def test_tracerecord_has_output_guard_field_default_none():
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    assert hasattr(rec, "output_guard")
    assert rec.output_guard is None


def test_tracerecord_output_guard_accepts_threats():
    """guards.output.sanitize_output 결과 — leak / profanity 라벨 contract."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    rec.output_guard = {"threats": ["output_leak:<|im_end|>", "profanity:시발"]}
    assert "output_leak:<|im_end|>" in rec.output_guard["threats"]


def test_tracerecord_no_schema_version_field():
    """schema_version 필드는 의도적으로 두지 않음 — 단일 producer 환경의 미니멀 정책."""
    from src.v1.rag.trace import TraceRecord
    rec = TraceRecord()
    assert not hasattr(rec, "schema_version")

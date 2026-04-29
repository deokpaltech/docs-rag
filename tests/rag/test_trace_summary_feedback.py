"""trace_summary._aggregate_feedback 핵심 로직 단위 테스트
(과거 feedback_summary.py에서 trace_summary.py로 통합 후에도 동일한 시그니처 유지).

DB·JSONL I/O는 테스트 대상 아님. join + 매칭률 + signal별 집계 정확성만 검증.
TDD 우선이 아니라 "가치 있는 곳만 테스트" 원칙 — feedback endpoint 자체는 thin CRUD라
스킵, 집계 로직은 잘못 계산하면 의사결정 오염되니 테스트.
"""
from __future__ import annotations


def test_summarize_empty():
    from scripts.trace_summary import _aggregate_feedback as summarize
    s = summarize([], {})
    assert s["period_total"] == 0
    assert s["matched_count"] == 0
    assert s["trace_match_rate"] == 0.0
    assert s["signal_distribution"] == {}


def test_summarize_signal_distribution():
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [
        {"trace_id": "t1", "signal": "up"},
        {"trace_id": "t2", "signal": "up"},
        {"trace_id": "t3", "signal": "down"},
        {"trace_id": "t4", "signal": "reformulated"},
    ]
    s = summarize(feedbacks, traces={})
    assert s["signal_distribution"] == {"up": 2, "down": 1, "reformulated": 1}


def test_summarize_match_rate_calculation():
    """trace 9/10 매칭 시 매칭률 0.9."""
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [{"trace_id": f"t{i}", "signal": "up"} for i in range(10)]
    traces = {f"t{i}": {} for i in range(9)}  # t0~t8만 매칭
    s = summarize(feedbacks, traces)
    assert s["matched_count"] == 9
    assert s["trace_match_rate"] == 0.9


def test_summarize_top1_score_by_signal():
    """signal별 rerank top-1 평균 — down은 일반적으로 낮은 score 패턴 확인용."""
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [
        {"trace_id": "t1", "signal": "up"},
        {"trace_id": "t2", "signal": "up"},
        {"trace_id": "t3", "signal": "down"},
    ]
    traces = {
        "t1": {"retrieval": {"rerank_scores": [0.9, 0.7]}},
        "t2": {"retrieval": {"rerank_scores": [0.8]}},
        "t3": {"retrieval": {"rerank_scores": [0.4]}},
    }
    s = summarize(feedbacks, traces)
    assert s["top1_score_by_signal"]["up"] == 0.85   # (0.9+0.8)/2
    assert s["top1_score_by_signal"]["down"] == 0.4


def test_summarize_risk_by_signal():
    """signal별 risk_level 집계 — hard_fail이 down과 상관 있는지 보는 용."""
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [
        {"trace_id": "t1", "signal": "up"},
        {"trace_id": "t2", "signal": "down"},
        {"trace_id": "t3", "signal": "down"},
    ]
    traces = {
        "t1": {"verification": {"risk_level": "pass"}},
        "t2": {"verification": {"risk_level": "hard_fail"}},
        "t3": {"verification": {"risk_level": "hard_fail"}},
    }
    s = summarize(feedbacks, traces)
    assert s["risk_by_signal"]["up"] == {"pass": 1}
    assert s["risk_by_signal"]["down"] == {"hard_fail": 2}


def test_summarize_trace_without_retrieval_or_verification():
    """trace에 retrieval/verification 필드 없어도 깨지지 않아야 (안전)."""
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [{"trace_id": "t1", "signal": "up"}]
    traces = {"t1": {}}  # 빈 trace
    s = summarize(feedbacks, traces)
    assert s["matched_count"] == 1
    assert s["top1_score_by_signal"] == {}
    assert s["risk_by_signal"] == {}


def test_summarize_unmatched_feedback_excluded_from_score_aggregation():
    """매칭 안 된 feedback은 score/risk 집계에서 제외 (당연하지만 회귀 방지)."""
    from scripts.trace_summary import _aggregate_feedback as summarize
    feedbacks = [
        {"trace_id": "matched", "signal": "up"},
        {"trace_id": "missing", "signal": "up"},
    ]
    traces = {
        "matched": {"retrieval": {"rerank_scores": [0.9]}},
    }
    s = summarize(feedbacks, traces)
    # 'missing'은 trace 없으니 score 집계에서 제외 → up 평균은 0.9 그대로
    assert s["top1_score_by_signal"]["up"] == 0.9


# ==============================================================================
# eval_ragas.map_score_to_signal — Faithfulness → signal 매핑 로직
# 실제 UI가 없는 현 단계에서 proxy feedback 생성 기준. 임계값이 바뀌면 과거 데이터의
# signal 분포가 달라지므로 회귀 방지.
# ==============================================================================


def test_map_score_to_signal_high_faithfulness_is_up():
    from scripts.eval_ragas import map_score_to_signal
    assert map_score_to_signal(0.9) == "up"
    assert map_score_to_signal(0.7) == "up"  # 경계값


def test_map_score_to_signal_mid_faithfulness_is_reformulated():
    from scripts.eval_ragas import map_score_to_signal
    assert map_score_to_signal(0.6) == "reformulated"
    assert map_score_to_signal(0.4) == "reformulated"  # 경계값


def test_map_score_to_signal_low_faithfulness_is_down():
    from scripts.eval_ragas import map_score_to_signal
    assert map_score_to_signal(0.3) == "down"
    assert map_score_to_signal(0.0) == "down"
    assert map_score_to_signal(0.399) == "down"

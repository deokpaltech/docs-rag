"""서빙 경로 trace 기록.

요청 단위로 JSONL trace를 append (FastAPI BackgroundTasks로 비동기 쓰기).
ContextVar로 request-scoped 전파 → 헬퍼 함수 signature 변경 없이 어디서든 업데이트 가능.
write 실패는 non-fatal (관측은 critical path 아님).
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "TraceRecord",
    "trace_record",
    "trace_span",
    "get_trace",
    "write_trace",
]


# schema_version 필드는 의도적으로 두지 않음 — producer가 본 서비스 단일이고 consumer도
# trace_summary.py 단일 스크립트(--feedback 옵션 포함)뿐이라 분산 환경의 schema 협상 비용이
# 안 듦. 신규 필드는 항상 optional 추가 + consumer는 dict.get(...) 으로 방어적 읽기.
# OpenTelemetry / Kafka 같은 분산 환경 이관 시 그쪽 schema_url로 위임 예정.
DEFAULT_BASE_DIR = Path("data/eval/trace")

_current: ContextVar["TraceRecord | None"] = ContextVar("current_trace", default=None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _default_decomposition() -> dict[str, Any]:
    return {"method": "none", "subqueries": []}


def _default_crag() -> dict[str, Any]:
    return {"retries": 0, "attempts": [], "score_before": None, "score_after": None}


@dataclass
class TraceRecord:
    """요청 1개당 1 레코드. ContextVar로 전파되어 파이프라인 전 구간에서 누적 업데이트."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_now_iso)
    endpoint: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    # Input Guard 결과 — PII 마스킹 메타데이터.
    # {"pii_found": ["RRN", "PHONE", ...], "pii_count": int}.
    # request 필드의 query/keywords는 이미 마스킹된 상태로 저장됨 (raw PII는 trace에 없음).
    input_guard: dict[str, Any] | None = None
    route: dict[str, Any] | None = None
    decomposition: dict[str, Any] = field(default_factory=_default_decomposition)
    crag: dict[str, Any] = field(default_factory=_default_crag)
    retrieval: dict[str, Any] | None = None
    sibling: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    # Critic dispatch 결과 — verification 후 failure_type 분류·regenerate 동작 기록.
    # invoked=False면 critic 미개입 (verification.risk_level이 pass/warn). 그 외 invoked=True와
    # failure_type / action_taken / before_risk / after_risk / regenerate_improved 채워짐.
    critic: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    # Output Guard 결과 — LLM 답변에서 검출된 role token leak / 욕설 라벨.
    # {"threats": ["output_leak:<|im_start|>", "profanity:..."]}. answer 필드는 이미 정제된 상태.
    output_guard: dict[str, Any] | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    error: dict[str, str] | None = None


@contextmanager
def trace_record(endpoint: str, request: dict[str, Any]) -> Iterator[TraceRecord]:
    """요청 진입점에서 한 번 호출. ContextVar 바인딩 후 자동 해제."""
    rec = TraceRecord(endpoint=endpoint, request=request)
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)


@contextmanager
def trace_span(name: str) -> Iterator[None]:
    """구간 시간 측정. trace 없으면 no-op, 있으면 누적 합산 (CRAG 재시도 대응)."""
    rec = _current.get()
    if rec is None:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        rec.timing_ms[name] = round(rec.timing_ms.get(name, 0) + elapsed_ms, 2)


def get_trace() -> "TraceRecord | None":
    """현재 컨텍스트의 record. trace_record 바깥이면 None."""
    return _current.get()


def write_trace(rec: TraceRecord, base_dir: Path = DEFAULT_BASE_DIR) -> None:
    """BackgroundTasks로 호출. 모든 예외 흡수 — 서빙 경로에 영향 없어야 함."""
    try:
        date_dir = base_dir / _today_utc()
        date_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(rec), ensure_ascii=False, default=str)
        with (date_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        # lazy import로 circular 회피 (rag/__init__.py가 trace를 re-export하므로)
        from ..logger import api_logger

        api_logger.error(f"trace write 실패 (무시): {exc}")

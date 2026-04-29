"""관측 인프라 DoD 자동 검증.

서비스가 `docker compose up -d` 상태여야 함.
11-step (endpoint health · trace 발행 · schema · aggregation · critic · input_guard)을
자동 체크하고, 실패 시 즉시 exit 1. 자동화 불가한 항목은 스크립트 끝에서 수동 가이드로 안내.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- 설정 ---------------------------------------------------------------------
API_BASE = "http://localhost:8002/api/v1/docs-rag"
DOCS_URL = "http://localhost:8002/docs"
TRACE_ROOT = Path("data/eval/trace")

TRACE_WRITE_POLL_TIMEOUT = 3.0       # BackgroundTask write 대기 최대 시간 (초)
TRACE_WRITE_POLL_INTERVAL = 0.2

# --- Canonical 테스트 쿼리 ----------------------------------------------------
CANONICAL_ANSWER_QUERY = {
    "query": "무면허운전 시 보험금 지급이 되나요?",
    "service_code": "01", "document_id": "0004", "top_k": 3,
}
CANONICAL_RETRIEVE_QUERY = {
    "query": "보험금 지급 조건",
    "service_code": "01", "document_id": "0004", "top_k": 5,
}
COMPARISON_QUERY = {
    "query": "1종과 2종의 차이가 뭔가요?",
    "service_code": "01", "document_id": "0001", "top_k": 3,
}

# --- 응답에 있으면 안 되는 필드 (slim projection 정책) ------------------------
BANNED_RESPONSE_FIELDS = (
    "grounded", "ungrounded_refs", "claims",
    "missing_refs", "numeric_mismatches", "review_required",
)

# --- 수동 DoD 항목 ------------------------------------------------------------
MANUAL_DOD_HINTS = (
    "#7  trace write 실패 시 서빙 영향 없음 — data/eval/trace/를 readonly로 만들고 /answer 호출",
    "#10 semantic_support 필드 코드 전체에서 0건 — grep -r semantic_support src/",
    "#14 docs 응답 예시가 실제 응답과 일치 — diff",
    "#17 새 Python 패키지 의존성 0건 — git diff pyproject.toml uv.lock",
)


# --- 로그 헬퍼 ----------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


# --- Trace 파일 I/O -----------------------------------------------------------

def _today_traces_path() -> Path:
    return TRACE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%d") / "traces.jsonl"


def _read_traces() -> list[dict]:
    fp = _today_traces_path()
    if not fp.exists():
        return []
    return [json.loads(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_trace_by_query(
    query: str,
    endpoint: str,
    timeout_s: float = TRACE_WRITE_POLL_TIMEOUT,
) -> dict:
    """BackgroundTask async write race 해소용 polling."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = [
            r for r in _read_traces()
            if r.get("request", {}).get("query") == query and r.get("endpoint") == endpoint
        ]
        if matches:
            return matches[-1]
        time.sleep(TRACE_WRITE_POLL_INTERVAL)
    _fail(f"trace not found within {timeout_s}s: query={query!r} endpoint={endpoint}")
    return {}   # unreachable — _fail exits. 타입 체커용.


# --- Steps --------------------------------------------------------------------

def step_1_health() -> None:
    r = requests.get(DOCS_URL, timeout=5)
    if r.status_code != 200:
        _fail(f"/docs returned {r.status_code}")
    _ok("step 1: API /docs reachable")


def step_2_answer_shape() -> None:
    """DoD #8, #9: /answer 응답 스키마 슬림 + 금지 필드 부재."""
    r = requests.post(f"{API_BASE}/answer", json=CANONICAL_ANSWER_QUERY, timeout=120)
    if r.status_code != 200:
        _fail(f"/answer returned {r.status_code}: {r.text[:200]}")
    resp = r.json()

    for k in ("query", "answer", "elapsed_ms", "sources", "route"):
        if k not in resp:
            _fail(f"/answer response missing key: {k}")

    verification = resp.get("verification")
    if verification is not None and set(verification.keys()) != {"risk_level", "warnings"}:
        _fail(f"/answer verification unexpected keys: {set(verification.keys())}")

    raw = json.dumps(resp, ensure_ascii=False)
    for banned in BANNED_RESPONSE_FIELDS:
        if f'"{banned}"' in raw:
            _fail(f"/answer response contains banned field: {banned!r}")

    _ok("step 2: /answer shape OK (verification slim, banned fields absent)")


def step_3_retrieve_shape() -> None:
    """/retrieve 응답 스키마."""
    r = requests.post(f"{API_BASE}/retrieve", json=CANONICAL_RETRIEVE_QUERY, timeout=60)
    if r.status_code != 200:
        _fail(f"/retrieve returned {r.status_code}: {r.text[:200]}")
    resp = r.json()
    for k in ("query", "total", "elapsed_ms", "sources"):
        if k not in resp:
            _fail(f"/retrieve response missing key: {k}")
    _ok("step 3: /retrieve shape OK")


def step_4_trace_written() -> None:
    """DoD #1: trace 파일 기록 (polling으로 async write 대기)."""
    deadline = time.time() + TRACE_WRITE_POLL_TIMEOUT
    while time.time() < deadline:
        traces = _read_traces()
        if len(traces) >= 2:
            _ok(f"step 4: trace file has {len(traces)} lines")
            return
        time.sleep(TRACE_WRITE_POLL_INTERVAL)
    _fail(f"expected >=2 trace lines, got {len(_read_traces())}")


def step_5_answer_fields() -> None:
    """DoD #2: answer trace에 7종 지표 populate."""
    rec = _find_trace_by_query(CANONICAL_ANSWER_QUERY["query"], endpoint="answer")
    for k in ("route", "retrieval", "verification", "crag", "decomposition", "timing_ms"):
        if rec.get(k) is None:
            _fail(f"trace[{k}] is null in answer record")
    for span in ("query_embed", "qdrant_search", "rerank", "llm_generate", "total"):
        if span not in rec["timing_ms"]:
            _fail(f"timing_ms missing span: {span}")
    _ok("step 5: answer trace has all 7 metric dimensions")


def step_6_retrieve_fields() -> None:
    """DoD #3: retrieve trace는 verification/answer null, crag.retries=0."""
    rec = _find_trace_by_query(CANONICAL_RETRIEVE_QUERY["query"], endpoint="retrieve")
    if rec.get("verification") is not None:
        _fail("retrieve trace verification should be null")
    if rec.get("answer") is not None:
        _fail("retrieve trace answer should be null")
    if rec["crag"]["retries"] != 0:
        _fail(f"retrieve crag.retries should be 0, got {rec['crag']['retries']}")
    _ok("step 6: retrieve trace has null verification/answer, crag.retries=0")


def step_7_comparison_decomposition() -> None:
    """DoD #5: COMPARISON 쿼리 → decomposition.method ∈ {rule, llm, llm_failed}."""
    r = requests.post(f"{API_BASE}/answer", json=COMPARISON_QUERY, timeout=120)
    if r.status_code != 200:
        _fail(f"COMPARISON /answer returned {r.status_code}")
    rec = _find_trace_by_query(COMPARISON_QUERY["query"], endpoint="answer")
    method = rec.get("decomposition", {}).get("method")
    if method not in ("rule", "llm", "llm_failed", "none"):
        _fail(f"decomposition.method unexpected: {method}")
    if method == "none":
        _fail("COMPARISON should trigger decomposition, got 'none'")
    _ok(f"step 7: COMPARISON decomposition.method={method}")


def step_8_crag_retry_if_any() -> None:
    """DoD #4: CRAG 재시도 attempts[] populate (warn-only — 트리거 여부는 쿼리 의존)."""
    retried = [r for r in _read_traces() if r.get("crag", {}).get("retries", 0) > 0]
    if not retried:
        _warn(
            "step 8: no CRAG retry triggered — DoD #4 unverified "
            "(초기 rerank score가 모두 threshold 이상)"
        )
        return
    sample = retried[0]
    if len(sample["crag"]["attempts"]) < 2:
        _fail(
            f"CRAG retries={sample['crag']['retries']} but "
            f"attempts has {len(sample['crag']['attempts'])}"
        )
    _ok(f"step 8: CRAG retry observed (n={len(retried)}), attempts array populated")


def step_9_aggregation() -> None:
    """trace_summary.py 실행 + JSON 생성 + 키 확인."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/trace_summary.py"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        _fail(f"trace_summary exit {result.returncode}: {result.stderr[-300:]}")

    section_count = result.stdout.count("───")
    if section_count < 9:
        _fail(f"expected >=9 sections, got {section_count}")

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    summary_fp = TRACE_ROOT.parent / f"trace_summary_{today}.json"
    if not summary_fp.exists():
        _fail(f"summary JSON not generated: {summary_fp}")

    required_keys = {
        "strategies", "query_types", "decomposition_methods",
        "rerank_top1", "crag", "risk_levels", "provenance",
        "latency_ms", "error_types",
    }
    actual_keys = set(json.loads(summary_fp.read_text(encoding="utf-8")).keys())
    missing = required_keys - actual_keys
    if missing:
        _fail(f"summary JSON missing keys: {missing}")
    _ok("step 9: aggregation script runs, summary JSON has required keys")


def step_10_critic_schema() -> None:
    """verification 있는 trace에 critic 필드 발행 확인 (schema_version 필드 없음 — Option B 정책)."""
    traces = _read_traces()
    if not traces:
        _warn("step 10: trace 없음 — critic 검증 skip")
        return

    for t in traces:
        if t.get("verification") is None:
            continue  # verification 없으면 critic도 없는 게 정상 (예: refusal)
        critic = t.get("critic")
        if critic is None:
            _fail(f"verification 있는 trace에 critic 누락 (trace_id={t.get('trace_id')})")
        if "invoked" not in critic:
            _fail(f"critic.invoked 키 없음 (trace_id={t.get('trace_id')})")
        if critic.get("invoked"):
            if not critic.get("failure_type"):
                _fail(f"critic invoked=True인데 failure_type 없음 (trace_id={t.get('trace_id')})")
            if not critic.get("action_taken"):
                _fail(f"critic invoked=True인데 action_taken 없음 (trace_id={t.get('trace_id')})")

    _ok("step 10: critic 필드 발행 확인")


def step_11_input_guard_schema() -> None:
    """모든 /retrieve·/answer trace에 input_guard 필드 발행 확인.

    PII 마스킹 발동 여부와 무관하게 메타데이터 dict는 항상 존재해야 함
    (pii_count=0이라도 {"pii_found": [], "pii_count": 0} 형태로 박혀야 가시성 보장).
    """
    traces = _read_traces()
    if not traces:
        _warn("step 11: trace 없음 — input_guard 검증 skip")
        return

    for t in traces:
        if t.get("endpoint") not in ("retrieve", "answer"):
            continue
        ig = t.get("input_guard")
        if ig is None:
            _fail(f"input_guard 필드 누락 (trace_id={t.get('trace_id')})")
        if "pii_found" not in ig or "pii_count" not in ig:
            _fail(f"input_guard.pii_found/pii_count 키 없음 (trace_id={t.get('trace_id')})")

    _ok("step 11: input_guard 필드 발행 확인")


STEPS = (
    step_1_health,
    step_2_answer_shape,
    step_3_retrieve_shape,
    step_4_trace_written,
    step_5_answer_fields,
    step_6_retrieve_fields,
    step_7_comparison_decomposition,
    step_8_crag_retry_if_any,
    step_9_aggregation,
    step_10_critic_schema,
    step_11_input_guard_schema,
)


# --- 메인 ---------------------------------------------------------------------

def main() -> None:
    print("=== DoD Smoke Test ===\n")
    for step in STEPS:
        step()
    print(f"\n=== Automated DoD ({len(STEPS)} steps): PASS ===")
    print("\n수동 확인 남은 4개:")
    for hint in MANUAL_DOD_HINTS:
        print(f"  {hint}")


if __name__ == "__main__":
    main()

"""인덱스 헬스 체크 스크립트.
Qdrant 컬렉션의 embedding space 품질을 Dispersion + Confusion Rate로 측정.
문서 추가/변경 후 주기적으로 실행하여 인덱스 오염 여부를 추적한다.

사용법:
    uv run python scripts/eval_index_health.py
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# Qdrant 연결 설정 (.env에서 읽거나 기본값)
# Makefile `include .env` + export 때문에 host에서도 QDRANT_HOST=qdrant가 오므로
# /.dockerenv 부재 시 host 실행으로 간주해 localhost로 override (feedback_summary.py와 동일 패턴).
if os.environ.get("QDRANT_HOST", "qdrant") == "qdrant" and not Path("/.dockerenv").exists():
    os.environ["QDRANT_HOST"] = "localhost"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", "6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "docs_rag_v1")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# RAG API
API_URL = os.environ.get("API_URL", "http://localhost:8002/api/v1/docs-rag")

# 출력 경로
EVAL_DIR = Path("data/eval/index")

# 임계값 (RAGRouter-Bench, 벤더 가이드 기반)
DISPERSION_THRESHOLD = 0.85     # 이상이면 벡터 뭉개짐 경고
CONFUSION_THRESHOLD = 0.15      # 이상이면 cross-domain 혼입 경고


# ── Qdrant REST API 헬퍼 ─────────────────────────────────────

def get_collection_info() -> dict:
    """컬렉션 기본 정보."""
    resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    resp.raise_for_status()
    return resp.json()["result"]


def count_by_document() -> dict[str, int]:
    """document_id별 벡터 수. scroll pagination으로 전체 순회."""
    counts = {}
    offset = None
    while True:
        body = {"limit": 10000, "with_payload": ["document_id"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json=body,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        points = result["points"]
        if not points:
            break
        for point in points:
            doc_id = point["payload"].get("document_id", "unknown")
            counts[doc_id] = counts.get(doc_id, 0) + 1
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return counts


def sample_vectors(sample_size: int = 500) -> list[dict]:
    """랜덤 샘플 벡터 추출 (dense 벡터 포함)."""
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={
            "limit": sample_size,
            "with_payload": ["document_id", "service_code", "heading_path"],
            "with_vector": ["dense"],
        },
    )
    resp.raise_for_status()
    return resp.json()["result"]["points"]


# ── 1. Dispersion 측정 ───────────────────────────────────────

def measure_dispersion(points: list[dict]) -> dict:
    """샘플 벡터의 평균 pairwise cosine similarity. 높을수록 벡터들이 뭉개져 있음."""
    vectors = []
    for p in points:
        vec = p.get("vector", {})
        if isinstance(vec, dict) and "dense" in vec:
            vectors.append(vec["dense"])
        elif isinstance(vec, list):
            vectors.append(vec)

    if len(vectors) < 10:
        return {"avg_cosine_sim": 0, "std": 0, "sample_size": len(vectors), "status": "insufficient"}

    vecs = np.array(vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs_norm = vecs / norms

    # 500개 벡터의 전체 pairwise는 ~125K쌍 → 1000쌍만 샘플링해도 통계적으로 충분.
    n = len(vecs_norm)
    num_pairs = min(1000, n * (n - 1) // 2)
    sims = []
    for _ in range(num_pairs):
        i, j = random.sample(range(n), 2)
        sim = float(np.dot(vecs_norm[i], vecs_norm[j]))
        sims.append(sim)

    avg = float(np.mean(sims))
    std = float(np.std(sims))
    status = "warning" if avg >= DISPERSION_THRESHOLD else "ok"

    return {"avg_cosine_sim": avg, "std": std, "sample_size": n, "num_pairs": num_pairs, "status": status}


# ── 2. Confusion Rate 측정 ───────────────────────────────────

# 테스트 쿼리: 각 쿼리가 어떤 document_id에서 결과가 나와야 하는지 태깅.
# document_id는 서버 DB 기준. 실행 전 count_by_document()로 확인 가능.
CONFUSION_QUERIES = [
    # document_id 매핑 (운영 DB 기준 — 자체 문서로 교체 시 갱신):
    #   0001 = 자녀보험 약관 (버전 A)
    #   0002 = 자녀보험 약관 (버전 B)
    #   0003 = 건강보험 약관
    #   0004 = 운전자상해보험 약관

    # 운전자상해보험 → 0004
    {"query": "무면허운전 시 보험금 지급", "expected_doc": "0004", "service_code": "01"},
    {"query": "운전면허정지 보장금 한도", "expected_doc": "0004", "service_code": "01"},
    {"query": "교통사고 벌금 지원", "expected_doc": "0004", "service_code": "01"},
    {"query": "자동차사고 변호사 선임비용", "expected_doc": "0004", "service_code": "01"},
    {"query": "이륜차 사용 위험 변경", "expected_doc": "0004", "service_code": "01"},

    # 자녀보험 → 0001 또는 0002 (같은 상품 다른 버전, 둘 다 허용)
    {"query": "태아 가입 시 선천이상 수술 보장", "expected_doc": "0001", "service_code": "01"},
    {"query": "1종과 2종의 차이", "expected_doc": "0001", "service_code": "01"},
    {"query": "어린이 골절 진단비", "expected_doc": "0001", "service_code": "01"},
    {"query": "자녀 입원일당 지급 기준", "expected_doc": "0001", "service_code": "01"},
    {"query": "신생아 질병 보장 범위", "expected_doc": "0001", "service_code": "01"},

    # 간편건강보험 → 0003
    {"query": "간편심사 고지 의무", "expected_doc": "0003", "service_code": "01"},
    {"query": "간편건강보험 해약환급금", "expected_doc": "0003", "service_code": "01"},
    {"query": "보험금을 지급하지 않는 사유", "expected_doc": "0003", "service_code": "01"},
]


# 같은 상품의 다른 버전은 동일 그룹으로 취급 (양쪽 다 relevant).
_DOC_GROUPS = {
    "0001": {"0001", "0002"},  # 자녀보험 — 같은 상품 두 버전, 양쪽 모두 정답으로 인정
    "0002": {"0001", "0002"},
}


def measure_confusion(top_k: int = 5) -> dict:
    """테스트 쿼리의 top-K 결과 중 엉뚱한 문서에서 나오는 비율."""
    total_results = 0
    confused_results = 0
    details = []

    for q in CONFUSION_QUERIES:
        try:
            resp = requests.post(
                f"{API_URL}/retrieve",
                json={"query": q["query"], "service_code": q["service_code"], "top_k": top_k},
                timeout=30,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            sources = data.get("sources", [])

            allowed = _DOC_GROUPS.get(q["expected_doc"], {q["expected_doc"]})
            total_in_query = len(sources)
            confused_in_query = sum(
                1 for s in sources
                if s.get("document_id") and s["document_id"] not in allowed
            )

            entry = {
                "query": q["query"],
                "expected_doc": q["expected_doc"],
                "result_count": total_in_query,
                "confused": confused_in_query,
                "got_docs": [s.get("document_id") for s in sources],
                "route": data.get("route", {}),
            }
            details.append(entry)
            total_results += total_in_query
            confused_results += confused_in_query

        except Exception as e:
            details.append({"query": q["query"], "error": str(e)})

    confusion_rate = confused_results / total_results if total_results > 0 else 0
    status = "warning" if confusion_rate >= CONFUSION_THRESHOLD else "ok"

    return {
        "confusion_rate": confusion_rate,
        "total_results": total_results,
        "confused_results": confused_results,
        "query_count": len(CONFUSION_QUERIES),
        "status": status,
        "details": details,
    }


# ── 메인 ─────────────────────────────────────────────────────

def main():
    print("=== 인덱스 헬스 체크 ===\n")

    # 컬렉션 기본 정보
    try:
        info = get_collection_info()
        total_points = info.get("points_count", 0)
        print(f"컬렉션: {COLLECTION} ({total_points:,} vectors)")
    except Exception as e:
        print(f"Qdrant 연결 실패: {e}")
        print(f"QDRANT_HOST={QDRANT_HOST}, QDRANT_PORT={QDRANT_PORT}")
        sys.exit(1)

    # 문서별 분포
    print("\n--- 문서별 벡터 분포 ---")
    doc_counts = count_by_document()
    for doc_id, count in sorted(doc_counts.items()):
        pct = count / total_points * 100 if total_points > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {doc_id}: {count:>6,} ({pct:4.1f}%) {bar}")

    # Dispersion
    print("\n--- Dispersion (벡터 공간 분산도) ---")
    print("  샘플 추출 중...")
    points = sample_vectors(500)
    dispersion = measure_dispersion(points)
    sim = dispersion["avg_cosine_sim"]
    status_icon = "⚠" if dispersion["status"] == "warning" else "✓"
    print(f"  평균 cosine similarity: {sim:.4f} (std={dispersion['std']:.4f})")
    print(f"  샘플: {dispersion['sample_size']}개, 페어: {dispersion.get('num_pairs', 0)}개")
    print(f"  판정: {status_icon} {'경고 — 벡터 뭉개짐 (>0.85)' if dispersion['status'] == 'warning' else '정상 — 분별력 유지'}")

    # Confusion Rate
    print("\n--- Confusion Rate (cross-document 혼입) ---")
    print(f"  테스트 쿼리 {len(CONFUSION_QUERIES)}개 실행 중...")
    confusion = measure_confusion()
    rate = confusion["confusion_rate"]
    status_icon = "⚠" if confusion["status"] == "warning" else "✓"
    print(f"  혼입률: {rate:.1%} ({confusion['confused_results']}/{confusion['total_results']})")
    print(f"  판정: {status_icon} {'경고 — cross-document 혼입 (>15%)' if confusion['status'] == 'warning' else '정상'}")

    # 규모 체크
    print("\n--- 규모 체크 ---")
    if total_points > 100000:
        print(f"  ⚠ {total_points:,} vectors — 10만 초과. HNSW latency 증가 가능.")
    else:
        print(f"  ✓ {total_points:,} vectors — 규모 적정.")

    # 요약
    print(f"\n{'='*50}")
    issues = []
    if dispersion["status"] == "warning":
        issues.append(f"Dispersion {sim:.4f} > {DISPERSION_THRESHOLD}")
    if confusion["status"] == "warning":
        issues.append(f"Confusion {rate:.1%} > {CONFUSION_THRESHOLD:.0%}")
    if total_points > 100000:
        issues.append(f"Scale {total_points:,} > 100K")

    if issues:
        print(f"  경고 {len(issues)}건: {', '.join(issues)}")
        print(f"  → 컬렉션 분리 검토 권장")
    else:
        print(f"  전체 정상. 현재 단일 컬렉션 유지 가능.")
    print(f"{'='*50}")

    # 결과 저장
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(),
        "collection": COLLECTION,
        "total_vectors": total_points,
        "doc_distribution": doc_counts,
        "dispersion": {k: v for k, v in dispersion.items() if k != "status"},
        "dispersion_status": dispersion["status"],
        "confusion": {k: v for k, v in confusion.items() if k not in ("status", "details")},
        "confusion_status": confusion["status"],
        "thresholds": {
            "dispersion": DISPERSION_THRESHOLD,
            "confusion": CONFUSION_THRESHOLD,
            "scale": 100000,
        },
    }
    out_path = EVAL_DIR / "index_health.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()

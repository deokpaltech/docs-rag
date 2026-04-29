"""검색 파이프라인 — 필터 빌드 / Hybrid (Dense+BM25+RRF) / 리랭킹 / 비교 분해 / 응답 포맷.

각 함수는 단일 책임 — `build_filter` 는 Qdrant Filter만, `search_rrf_only` 는 검색만,
`search_and_rerank` 는 검색+리랭킹, `search_comparison` 은 분해+합산+리랭킹 orchestration.
LLM helper(`rewrite_query`, `decompose_query_llm`) 도 검색 context에서만 쓰이므로 본 모듈에 동거.
"""
from __future__ import annotations

from qdrant_client.models import (
    Document as QdrantDocument,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchText,
    MatchValue,
    Prefetch,
    QuantizationSearchParams,
    Range,
    SearchParams,
)

from ..config import BM25_CONFIG, QDRANT_CONFIG
from ..config.settings import SEARCH_PREFETCH_MULTIPLIER
from ..logger import api_logger
from ..utils import embed_query
from .classifier import decompose_comparison
from .clients import invoke_clean, qdrant, reranker
from .prompts import DECOMPOSE_PROMPT, REWRITE_PROMPT
from .trace import get_trace, trace_span


def build_filter(
    service_code: str | None = None,
    document_id: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> Filter | None:
    """Qdrant Filter 생성. 모든 필터 None이면 None 반환."""
    must = []
    if service_code:
        must.append(FieldCondition(key="service_code", match=MatchValue(value=service_code)))
    if document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    if start_page is not None:
        must.append(FieldCondition(key="page_range[0]", range=Range(gte=start_page)))
    if end_page is not None:
        must.append(FieldCondition(key="page_range[1]", range=Range(lte=end_page)))
    if include_keywords:
        for kw in include_keywords:
            must.append(FieldCondition(key="content", match=MatchText(text=kw)))

    must_not = [FieldCondition(key="content", match=MatchText(text=kw)) for kw in (exclude_keywords or [])]

    if must or must_not:
        return Filter(must=must, must_not=must_not if must_not else None)
    return None


def search_rrf_only(
    query: str,
    top_k: int,
    query_filter: Filter | None = None,
    dense_factor: int = 6,
    bm25_factor: int = 6,
) -> list:
    """Hybrid 검색 — Dense (BGE-M3) + BM25 → RRF 융합. 리랭킹 없이 RRF score로 반환."""
    with trace_span("query_embed"):
        query_vector = embed_query(query)
    with trace_span("qdrant_search"):
        results = qdrant.query_points(
            collection_name=QDRANT_CONFIG["collection_name"],
            prefetch=[
                Prefetch(query=query_vector, using="dense",
                         limit=top_k * dense_factor, filter=query_filter),
                Prefetch(query=QdrantDocument(text=query, model="Qdrant/bm25"),
                         using=BM25_CONFIG["sparse_vector_name"],
                         limit=top_k * bm25_factor, filter=query_filter),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k * SEARCH_PREFETCH_MULTIPLIER,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=QDRANT_CONFIG["hnsw_ef"],
                quantization=QuantizationSearchParams(rescore=True, oversampling=2.0),
            ),
        )
    return results.points if results.points else []


def search_and_rerank(
    query: str,
    top_k: int,
    query_filter: Filter | None = None,
    dense_factor: int = 6,
    bm25_factor: int = 6,
):
    """Hybrid 검색 + CrossEncoder 리랭킹. dense_factor/bm25_factor로 후보 풀 비중 조절."""
    points = search_rrf_only(query, top_k, query_filter, dense_factor, bm25_factor)
    if not points:
        return []

    pairs = [(query, r.payload.get("content", "")) for r in points]
    with trace_span("rerank"):
        scores = reranker.predict(pairs)
    return sorted(zip(points, scores), key=lambda x: x[1], reverse=True)[:top_k]


def rewrite_query(query: str) -> str:
    """LLM으로 쿼리 재작성 — CRAG 루프에서 검색 품질 낮을 때 호출."""
    rewritten = invoke_clean(REWRITE_PROMPT.format_messages(query=query))
    api_logger.info(f"CRAG 쿼리 재작성: '{query}' → '{rewritten}'")
    return rewritten


def decompose_query_llm(query: str) -> list[str] | None:
    """LLM으로 비교 질문을 서브쿼리로 분해. SINGLE이면 None.
    "MULTI: [쿼리1, 쿼리2]" 형식 파싱 — 실패 시 None.
    """
    text = invoke_clean(DECOMPOSE_PROMPT.format_messages(query=query))
    if not text.startswith("MULTI:"):
        return None
    bracket = text.find("[")
    if bracket == -1:
        return None
    inner = text[bracket + 1:text.rfind("]")]
    queries = [q.strip().strip("'\"") for q in inner.split(",") if q.strip()]
    if len(queries) < 2:
        return None
    api_logger.info(f"LLM 쿼리 분해: '{query}' → {queries}")
    return queries


def search_comparison(query: str, top_k: int, query_filter, route) -> list:
    """비교 질문용 검색 — 서브쿼리 분해 → 각각 RRF 검색 → 합산 → 원 쿼리로 1회 리랭킹.
    1차 규칙(decompose_comparison) → 2차 LLM(decompose_query_llm) → 실패 시 단일 검색 폴백.
    first-wins: 초기 호출에서만 trace.decomposition 기록 (CRAG 재작성이 덮어쓰지 않게).
    """
    rec = get_trace()
    record_decomp = rec is not None and (rec.decomposition or {}).get("method") == "none"

    # 1차 규칙 → 2차 LLM → 폴백
    sub_queries = decompose_comparison(query)
    if sub_queries:
        if record_decomp:
            rec.decomposition = {"method": "rule", "subqueries": sub_queries}
    else:
        sub_queries = decompose_query_llm(query)
        if sub_queries and record_decomp:
            rec.decomposition = {"method": "llm", "subqueries": sub_queries}

    if not sub_queries:
        if record_decomp:
            rec.decomposition = {"method": "llm_failed", "subqueries": []}
        api_logger.info("비교 질문 분해 실패, 단일 검색 폴백")
        return search_and_rerank(
            query, top_k, query_filter,
            dense_factor=route.dense_factor, bm25_factor=route.bm25_factor,
        )

    api_logger.info(f"비교 질문 서브쿼리: {sub_queries}")

    # 서브쿼리별 RRF만 (리랭킹 생략) → 합산 → 원 쿼리로 1회 리랭킹
    seen_ids: set = set()
    all_points = []
    per_query_k = max(top_k // len(sub_queries), 3)

    for sq in sub_queries:
        for point in search_rrf_only(
            sq, per_query_k, query_filter,
            dense_factor=route.dense_factor, bm25_factor=route.bm25_factor,
        ):
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                all_points.append(point)

    if not all_points:
        return []

    pairs = [(query, p.payload.get("content", "")) for p in all_points]
    with trace_span("rerank"):
        scores = reranker.predict(pairs)
    return sorted(zip(all_points, scores), key=lambda x: x[1], reverse=True)[:top_k]


def format_sources(ranked: list) -> list[dict]:
    """Qdrant point + rerank score → API 응답용 dict 리스트.

    chunk_id는 응답의 citations[].supported_by_chunks가 sources를 lookup하는 키 —
    클라이언트가 inline citation UI 만들 때 매핑 인덱싱용.
    """
    sources = []
    for r, s in ranked:
        item = {
            "chunk_id": str(r.id),
            "document_id": r.payload.get("document_id"),
            "page_range": r.payload.get("page_range"),
            "content": r.payload.get("content"),
            "chunk_type": r.payload.get("chunk_type"),
            "rrf_score": round(r.score, 4),
            "rerank_score": round(float(s), 4),
        }
        image_paths = r.payload.get("image_paths")
        if image_paths:
            item["image_paths"] = image_paths
        sources.append(item)
    return sources

"""Sibling 복원 — 검색 hit 청크의 heading_path 안에서 ±SIBLING_WINDOW part를 합쳐 LLM context 구성.

검색은 작게(top_k=3~10), LLM 전달은 크게(같은 조항 ±N part). N+1 쿼리 방지 위해
heading_path들을 한 번의 Qdrant scroll로 배치 조회.

진입점은 `expand_siblings(ranked)` 하나. qdrant client는 clients.py 싱글톤 사용.
"""
from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from ..config import QDRANT_CONFIG
from ..config.settings import SIBLING_WINDOW
from .clients import qdrant


def _with_heading(heading_path: str, content: str) -> str:
    if heading_path:
        return f"{heading_path}\n\n{content}"
    return content


def _fetch_siblings_batch(sibling_requests: list[dict]) -> dict[str, list]:
    """여러 heading_path의 sibling을 1회 Qdrant scroll로.
    sibling_requests: [{"heading_path": str, "doc_id": str, "part_total": int}, ...]
    """
    if not sibling_requests:
        return {}

    hp_conditions = [
        FieldCondition(key="heading_path", match=MatchValue(value=req["heading_path"]))
        for req in sibling_requests
    ]
    # 모든 doc_id 동일 시 단일 필터로 좁힘 (성능). 다양 시 헤딩 OR로 충분.
    doc_ids = {req["doc_id"] for req in sibling_requests}
    total_limit = sum(req["part_total"] + 5 for req in sibling_requests)

    must_conditions = []
    if len(doc_ids) == 1:
        must_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=doc_ids.pop())))

    sibling_filter = Filter(must=must_conditions, should=hp_conditions)
    all_siblings = qdrant.scroll(
        collection_name=QDRANT_CONFIG["collection_name"],
        scroll_filter=sibling_filter,
        limit=min(total_limit, 500),
        with_payload=True,
    )[0]

    result: dict[str, list] = {}
    for sib in all_siblings:
        hp = sib.payload.get("heading_path") or ""
        result.setdefault(hp, []).append(sib)
    return result


def _select_window(siblings: list, hit_part: int, part_total: int) -> list[tuple[int, str]]:
    """hit 기준 ±SIBLING_WINDOW 범위의 sibling만 (part_index, content)."""
    min_part = max(1, hit_part - SIBLING_WINDOW)
    max_part = min(part_total, hit_part + SIBLING_WINDOW)
    result = []
    for sib in sorted(siblings, key=lambda s: s.payload.get("part_index") or 0):
        pi = sib.payload.get("part_index") or 0
        if min_part <= pi <= max_part:
            result.append((pi, sib.payload.get("content", "")))
    return result


def _group_sections(sections: list[tuple[str, int, str]]) -> str:
    """heading_path별로 content를 합쳐 마크다운으로 직렬화."""
    sections.sort(key=lambda x: (x[0], x[1]))
    result_parts: list[str] = []
    current_path = None
    group_contents: list[str] = []
    for hp, _, content in sections:
        if hp != current_path:
            if group_contents:
                result_parts.append(_with_heading(current_path or "", "\n\n".join(group_contents)))
            current_path = hp
            group_contents = [content]
        else:
            group_contents.append(content)
    if group_contents:
        result_parts.append(_with_heading(current_path or "", "\n\n".join(group_contents)))
    return "\n\n---\n\n".join(result_parts)


def expand_siblings(ranked: list) -> str:
    """Sibling 복원 진입점. hit된 청크의 heading_path로 같은 조항의 나머지 part를 가져와 합침.
    배치 조회로 N+1 쿼리 회피.
    """
    seen_paths: set[str] = set()
    seen_keys: set[tuple[str, int]] = set()
    expanded_sections: list[tuple[str, int, str]] = []

    sibling_requests = []
    single_parts = []
    no_heading = []
    multi_parts = []

    for r, _ in ranked:
        heading_path = r.payload.get("heading_path") or ""
        if not heading_path:
            no_heading.append(r)
            continue
        if heading_path in seen_paths:
            continue
        seen_paths.add(heading_path)

        part_total = r.payload.get("part_total") or 1
        if part_total <= 1:
            single_parts.append((heading_path, r))
        else:
            multi_parts.append((heading_path, r))
            sibling_requests.append({
                "heading_path": heading_path,
                "doc_id": r.payload.get("document_id"),
                "part_total": part_total,
            })

    siblings_by_hp = _fetch_siblings_batch(sibling_requests) if sibling_requests else {}

    for r in no_heading:
        expanded_sections.append(("", 0, r.payload.get("content", "")))

    for heading_path, r in single_parts:
        expanded_sections.append((heading_path, 1, r.payload.get("content", "")))
        seen_keys.add((heading_path, 1))

    for heading_path, r in multi_parts:
        hit_part = r.payload.get("part_index") or 1
        part_total = r.payload.get("part_total") or 1
        siblings = siblings_by_hp.get(heading_path, [])
        for pi, content in _select_window(siblings, hit_part, part_total):
            key = (heading_path, pi)
            if key not in seen_keys:
                seen_keys.add(key)
                expanded_sections.append((heading_path, pi, content))

    return _group_sections(expanded_sections)

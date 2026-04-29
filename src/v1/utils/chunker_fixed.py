"""Fixed-size 윈도우 슬라이딩 청킹.
문서 전체를 구조 무시하고 고정 크기로 슬라이딩.
메타데이터에 가장 가까운 이전 헤딩을 상속 (content는 본문만).
"""

from __future__ import annotations

import re
import logging

from .preprocess import normalize_whitespace, clean_text, remove_toc, is_page_marker, parse_heading
from .chunker_adaptive import Chunk
from ..config import FIXED_WINDOW_SIZE, FIXED_OVERLAP_SIZE

WINDOW_SIZE = FIXED_WINDOW_SIZE
OVERLAP_SIZE = FIXED_OVERLAP_SIZE

log = logging.getLogger(__name__)


def _build_page_index(text: str) -> list[tuple[int, int]]:
    """원문에서 (offset, page_number) 인덱스 구축."""
    entries = []
    offset = 0
    for line in text.split('\n'):
        p = is_page_marker(line)
        if p is not None:
            entries.append((offset, p))
        offset += len(line) + 1
    return entries


def _build_heading_index(text: str) -> list[tuple[int, str, list[str]]]:
    """원문에서 (offset, heading, heading_path) 인덱스 구축.
    각 헤딩의 offset과 계층 경로를 기록."""
    entries: list[tuple[int, str, list[str]]] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    offset = 0

    for line in text.split('\n'):
        parsed = parse_heading(line)
        if parsed:
            level, heading_text = parsed
            # 현재 레벨 이상의 스택 항목 제거 (상위 헤딩으로 복귀)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            path = [h for _, h in stack]
            entries.append((offset, heading_text, path))
        offset += len(line) + 1

    return entries


def _find_heading_at_position(heading_index: list, position: int) -> tuple[str | None, list[str]]:
    """position 이전의 가장 가까운 헤딩과 경로."""
    heading = None
    path: list[str] = []
    for offset, h, p in heading_index:
        if offset > position:
            break
        heading = h
        path = p
    return heading, path


def _find_page_at_position(page_index: list, position: int) -> int | None:
    """position 이전의 가장 가까운 페이지 번호."""
    page = None
    for offset, p in page_index:
        if offset > position:
            break
        page = p
    return page


def _sliding_window(text: str, window_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """고정 크기 윈도우 슬라이딩. (chunk_text, start_pos, end_pos) 반환."""
    if len(text) <= window_size:
        return [(text, 0, len(text))]

    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunk = text[start:end]

        # 단어 중간 절단 방지. 윈도우 후반부에서 가장 가까운 경계를 찾음.
        if end < len(text):
            last_break = max(chunk.rfind('\n'), chunk.rfind(' '), chunk.rfind('.'))
            if last_break > window_size * 0.5:
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append((chunk, start, end))

        step = end - start - overlap
        if step <= 0:
            step = window_size // 2
        start = start + step

    return chunks


# 메인
def chunk_markdown(text: str, source_file: str = "", service_code: str = "") -> list[Chunk]:
    """고정 크기 윈도우 슬라이딩 청킹. 전처리: 정규화 → TOC 제거 → 노이즈 제거."""
    raw_text = text
    normalized = normalize_whitespace(text)
    normalized = remove_toc(normalized)
    cleaned = clean_text(normalized)

    if not cleaned:
        return []

    page_index = _build_page_index(raw_text)
    heading_index = _build_heading_index(normalized)
    windows = _sliding_window(cleaned, WINDOW_SIZE, OVERLAP_SIZE)

    log.info(f"고정 청킹: {len(cleaned):,}자 → {len(windows)}개 ({WINDOW_SIZE}자, 오버랩 {OVERLAP_SIZE}자)")

    # clean_text가 노이즈를 제거하면서 길이가 줄어드므로, 원문 offset 추정에 비율 매핑 사용.
    raw_len = len(normalized)
    cleaned_len = len(cleaned)
    ratio = raw_len / cleaned_len if cleaned_len > 0 else 1.0

    chunks = []
    for i, (chunk_text, start_pos, end_pos) in enumerate(windows):
        raw_start = int(start_pos * ratio)
        raw_end = int(end_pos * ratio)

        start_page = _find_page_at_position(page_index, raw_start)
        end_page = _find_page_at_position(page_index, raw_end)
        heading, heading_path = _find_heading_at_position(heading_index, raw_start)

        chunks.append(Chunk(
            content=chunk_text,
            heading=heading,
            metadata={
                "service_code": service_code,
                "source_file": source_file,
                "heading_path": heading_path,
                "page_start": start_page,
                "page_end": end_page,
            },
        ))

    return chunks

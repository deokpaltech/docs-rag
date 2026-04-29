"""PaddleOCR HTTP 클라이언트.
paddle 서비스(PP-StructureV3)에 HTTP로 OCR 요청을 보내고 결과를 파싱한다.
ODL과 동일한 패턴: Docker 서비스 → HTTP API → 클라이언트.
"""

import os
import re
from pathlib import Path

import requests

from ..config import (
    OCR_MIN_FILE_SIZE, OCR_MIN_IMAGE_WIDTH, OCR_MIN_IMAGE_HEIGHT,
    OCR_FIGURE_MIN_WIDTH, OCR_FIGURE_MIN_HEIGHT,
    OCR_MAX_ASPECT_RATIO, OCR_MAX_IMAGE_DIMENSION,
    OCR_MIN_PIXEL_STDDEV,
    OCR_MIN_TEXT_LENGTH, DATA_DIR,
)

PADDLE_URL = os.environ.get("PADDLE_URL", "http://paddle:5003")

_SPECIAL_ONLY_RE = re.compile(r'^[\s\-–—_·…=|/\\:;,.!?@#$%^&*(){}[\]<>]+$')
# 수식 잔해 패턴
_MATH_NOISE_RE = re.compile(r'^[x_{}()\-\d\s\^]+$')
# CJK 한자 범위 (한글이 아닌 중국어 한자)
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_HANGUL_RE = re.compile(r'[가-힣]')
_ALPHA_RE = re.compile(r'[a-zA-Z]')


def _to_container_path(host_path: Path) -> str:
    """호스트 경로를 paddle 컨테이너 경로로 변환 (./data → /data)."""
    rel = host_path.relative_to(DATA_DIR)
    return f"/data/{rel}"


def get_image_info(image_path: Path) -> dict:
    """이미지 메타데이터 추출.
    차원 외에 픽셀 표준편차(stddev)도 계산해 단색/투명 마스크 판별에 사용.
    """
    info = {"width": 0, "height": 0, "file_size": 0, "exists": False, "stddev": 0.0}
    if not image_path.exists():
        return info
    info["exists"] = True
    info["file_size"] = image_path.stat().st_size
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            info["height"], info["width"] = img.shape[:2]
            # 전체 픽셀 표준편차 (BGR 3채널 평균). 단색 이미지는 ≈0.
            try:
                info["stddev"] = float(img.std())
            except Exception:
                info["stddev"] = 0.0
    except ImportError:
        pass
    return info


def is_valid_image(info: dict) -> tuple[bool, str]:
    """입구 필터 6단계. ODL의 external 모드가 PDF 내부 XObject를 전량 떨구므로
    여기서 "의미 있는 figure/chart/table 후보"만 paddle로 넘긴다.

      L1 file_size < OCR_MIN_FILE_SIZE      (빈 PNG, 투명 오버레이)
      L2 w·h < OCR_MIN_IMAGE_WIDTH/HEIGHT    (구분선 잔해)
      L3 w < OCR_FIGURE_MIN_WIDTH or
         h < OCR_FIGURE_MIN_HEIGHT           (아이콘/로고/QR)
      L4 ratio ≥ OCR_MAX_ASPECT_RATIO        (가로띠/세로띠)
      L5 w·h > OCR_MAX_IMAGE_DIMENSION       (대형 배너, 리사이즈 오버헤드)
      L6 stddev < OCR_MIN_PIXEL_STDDEV       (단색/투명 마스크)

    반환: (통과 여부, 실패 사유 태그)
    """
    if not info["exists"]:
        return False, "not_exists"
    if info.get("file_size", 0) < OCR_MIN_FILE_SIZE:
        return False, "file_size"

    w, h = info["width"], info["height"]
    if w < OCR_MIN_IMAGE_WIDTH or h < OCR_MIN_IMAGE_HEIGHT:
        return False, "too_small"
    # figure는 가로·세로 모두 일정 크기 이상. 한쪽이라도 미만이면 썸네일/로고/아이콘.
    if w < OCR_FIGURE_MIN_WIDTH or h < OCR_FIGURE_MIN_HEIGHT:
        return False, "icon_size"
    if h > 0 and w / h >= OCR_MAX_ASPECT_RATIO:
        return False, "aspect_h"
    if w > 0 and h / w >= OCR_MAX_ASPECT_RATIO:
        return False, "aspect_v"
    if w > OCR_MAX_IMAGE_DIMENSION or h > OCR_MAX_IMAGE_DIMENSION:
        return False, "too_large"
    # stddev=0은 측정 실패(cv2 ImportError 등) → 통과 허용
    if OCR_MIN_PIXEL_STDDEV > 0 and info.get("stddev", 0) > 0:
        if info["stddev"] < OCR_MIN_PIXEL_STDDEV:
            return False, "flat_color"

    return True, "ok"


def is_meaningful_ocr_result(text: str) -> bool:
    """출구 필터: 길이, 특수문자 비율, 수식 잔해, 한글+영문 비율 30% 미만 체크."""
    stripped = text.strip()
    if not stripped or len(stripped) < OCR_MIN_TEXT_LENGTH:
        return False
    if _SPECIAL_ONLY_RE.match(stripped):
        return False
    if _MATH_NOISE_RE.match(stripped):
        return False
    hangul_count = len(_HANGUL_RE.findall(stripped))
    alpha_count = len(_ALPHA_RE.findall(stripped))
    total_chars = len(stripped.replace(" ", "").replace("\n", ""))
    if total_chars > 0 and (hangul_count + alpha_count) / total_chars < 0.3:
        return False
    return True


def ocr_image(image_path: Path) -> dict:
    """paddle /ocr 호출 → 블록 분류 결과 반환.

    Returns:
      {
        "text":    콘텐츠 블록(title/text/caption/formula 등) 합친 문자열,
        "tables":  table 블록을 마크다운으로 변환한 리스트 (각각 별도 table 청크),
        "dropped": drop 라벨 목록 (header/footer/page_number, 디버깅용),
      }
    """
    container_path = _to_container_path(image_path)
    empty = {"text": "", "tables": [], "dropped": []}

    try:
        # timeout=300s: CPU 모드 + paddle 내부 lock 직렬화로 4번째 동시 요청이
        # 앞 3개 대기 포함 100초+ 걸릴 수 있음. 단독 호출은 ~20초.
        # timeout된 이미지는 빈 결과 반환 → ocr.py가 individual skip + task는 계속 진행.
        # 입구 필터(settings.py OCR_FIGURE_MIN_*)로 부수 이미지 컷되면 timeout 발생 자체 거의 없음.
        resp = requests.post(
            f"{PADDLE_URL}/ocr",
            json={"image_path": container_path},
            timeout=300,
        )
    except requests.ConnectionError:
        return empty

    if resp.status_code != 200:
        return empty

    data = resp.json()
    blocks = data.get("blocks", [])

    text_parts: list[str] = []
    tables: list[str] = []
    dropped: list[str] = []

    for block in blocks:
        block_type = block.get("type", "")
        text = (block.get("text") or "").strip()
        html = block.get("html") or ""
        label = block.get("_label", "")

        if block_type == "drop":
            # header/footer/page_number — 청킹 제외, 디버깅 로그용으로만 기록
            if label:
                dropped.append(label)
            continue

        if block_type == "table":
            # 표는 별도 청크로 분리
            md = _html_table_to_markdown(html) if html else text
            if md and md.strip():
                tables.append(md.strip())
            continue

        # 콘텐츠 블록 (text/title/paragraph_title/doc_title/caption/formula 등)
        if text:
            text_parts.append(text)

    return {
        "text": "\n\n".join(text_parts),
        "tables": tables,
        "dropped": dropped,
    }


def _html_table_to_markdown(html: str) -> str:
    """HTML 테이블을 Markdown 테이블로 변환."""
    try:
        from html.parser import HTMLParser

        rows = []
        current_row = []
        current_cell = ""
        in_cell = False

        class TableParser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                nonlocal in_cell, current_cell
                if tag in ("td", "th"):
                    in_cell = True
                    current_cell = ""

            def handle_endtag(self, tag):
                nonlocal in_cell, current_row
                if tag in ("td", "th"):
                    in_cell = False
                    current_row.append(current_cell.strip())
                elif tag == "tr":
                    if current_row:
                        rows.append(current_row[:])
                        current_row.clear()

            def handle_data(self, data):
                nonlocal current_cell
                if in_cell:
                    current_cell += data

        parser = TableParser()
        parser.feed(html)

        if not rows:
            return html

        col_count = max(len(r) for r in rows)
        lines = []
        for i, row in enumerate(rows):
            padded = row + [""] * (col_count - len(row))
            lines.append("| " + " | ".join(padded) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * col_count) + " |")

        return "\n".join(lines)
    except Exception:
        return html

"""설정값 관리. 검색/청킹/OCR 상수를 일원화. 하드코딩 금지."""

import os
from pathlib import Path

# __file__ 기준 자동 inference — Docker(/app/src/v1/config/...)·호스트 둘 다 .../data로 해석.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
OUTPUT_RAW_DIR = DATA_DIR / "output" / "raw"
OUTPUT_PROCESSED_DIR = DATA_DIR / "output" / "processed"
FINISHED_DIR = DATA_DIR / "finished"
ERROR_DIR = DATA_DIR / "error"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "model"

# 청크 — adaptive (헤딩 트리 기반)
# BGE-M3 sweet spot: 256~512 tokens ≈ 800~1600 chars (한국어 기준)
TEXT_MAX_CHARS = 1200       # 텍스트 전용 
TABLE_MAX_CHARS = 2400      # 표 — 행/열 보존 우선, 분할하지 않음 
CHUNK_MIN_CHARS = 300       # 이하 자투리는 이전 청크에 병합 

# 청크 — fixed (윈도우 슬라이딩)
FIXED_WINDOW_SIZE = 800
FIXED_OVERLAP_SIZE = 150

# 검색 — 하이브리드 검색 + sibling 복원 + CRAG
# SEARCH_PREFETCH_MULTIPLIER: RRF 융합 **후** 리랭킹 대상으로 가져올 후보 수 배수.
#   최종 리랭킹 대상 = top_k × SEARCH_PREFETCH_MULTIPLIER
#   예: top_k=10, 배수=3 → 30개를 CrossEncoder가 리랭킹 → 상위 10개 반환.
#   (이것과 구분: rag/router.py의 dense_factor/bm25_factor는 각 엔진이 RRF 풀에
#    밀어넣는 후보 수 배수. SEARCH_PREFETCH_MULTIPLIER는 RRF 이후의 리랭커 입력 크기.)
# 올리면 recall↑ · latency↑ (리랭커 호출 ∝ N). 경험적으로 3~5가 적정.
SEARCH_PREFETCH_MULTIPLIER = 3
# 검색은 작게(단일 청크), LLM에는 크게(sibling 포함) 전달하는 전략.
# ±2이면 hit 포함 최대 5개 청크가 하나의 context 블록. 약관 한 조문이 보통 3~5 part.
SIBLING_WINDOW = 2
# CrossEncoder rerank score 최소 기준. 운전자보험 약관 28개 질문 기준 0.3에서
# 재검색 트리거 비율이 ~15%로 적정. 올리면 재검색 빈번(latency↑), 내리면 저품질 허용.
CRAG_SCORE_THRESHOLD = 0.3
# CRAG 재검색 최대 횟수. 1회당 +2~3초(vLLM 재작성 + 재검색 + 리랭킹).
CRAG_MAX_RETRIES = 2

# OCR — PP-Structure
# 입구 필터 1: 파일 크기 < 100B면 1~2픽셀 PNG 확정 (빈 XObject), cv2 디코딩 전 차단.
OCR_MIN_FILE_SIZE = 100
# 입구 필터 2: 최소 차원. PDF 추출 시 1~2px 아티팩트 차단 (보수적 하한).
OCR_MIN_IMAGE_WIDTH = 10
OCR_MIN_IMAGE_HEIGHT = 10
# 입구 필터 3: figure/차트/표로 보기 어려운 작은 아이콘/로고/QR/페이지번호 크기 상한.
# 보험 약관 14개 처리 후 timeout 분석 결과 — 헤더/푸터/로고가 paddle 큐 적체의 주범.
# 400x300 미만은 의미 있는 figure보다는 장식/로고/페이지번호. 둘 다 미만일 때 차단.
OCR_FIGURE_MIN_WIDTH = 400
OCR_FIGURE_MIN_HEIGHT = 300
# 입구 필터 4: 가로띠/세로띠 비율 상한. 페이지 상/하단 구분선 차단.
OCR_MAX_ASPECT_RATIO = 10
# 입구 필터 5: 극단적 대형 이미지 상한. Paddle 내부 리사이즈 오버헤드 방지.
OCR_MAX_IMAGE_DIMENSION = 7000
# 입구 필터 6: 픽셀 표준편차 하한. 단색/투명 마스크/거의 빈 이미지 차단.
# 단색 ≈ 0, 페이지번호만 있는 거의 빈 이미지 < 10, 진짜 figure는 수십 이상.
# 5 → 10으로 상향 (페이지번호·도장 같은 부수 컷 강화). 0으로 설정 시 필터 비활성화.
OCR_MIN_PIXEL_STDDEV = 10.0

# 출구 필터: OCR 결과가 10자 미만이면 의미 있는 텍스트일 가능성 낮음.
# 운전자보험 기준 제거 대상의 65%가 too_short. false positive(유의미 텍스트 오제거) 낮음.
OCR_MIN_TEXT_LENGTH = 10


class StatusCode:
    """상태 코드"""
    WAITING = "00"
    COMPLETE_ALL = "11"
    COMPLETE_PDF_EXTRACT = "21"
    PROCESSING_PDF_EXTRACT = "22"
    COMPLETE_OCR = "23"
    PROCESSING_OCR = "24"
    COMPLETE_PREPROCESS = "31"
    PROCESSING_PREPROCESS = "32"
    COMPLETE_EMBED_VECTOR = "41"   # 벡터DB 적재/인덱싱 완료 (검색 가능 상태)
    PROCESSING_EMBED = "42"        # 임베딩 계산 중
    COMPLETE_EMBED = "43"          # 임베딩 계산 완료 (벡터DB 적재 전)
    ERROR_PDF_EXTRACT = "91"
    ERROR_OCR = "92"
    ERROR_PREPROCESS = "93"
    ERROR_PREPROCESS_DB = "94"
    ERROR_EMBED = "95"
    ERROR_EMBED_DB = "96"
    ERROR_ETC = "99"

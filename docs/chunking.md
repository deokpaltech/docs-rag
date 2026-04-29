# 청킹 전략

## 전략 개요

`.env`의 `CHUNKER_TYPE`으로 전략 선택. 같은 DB/인덱스에 공존 가능.

| | Adaptive | Fixed |
|---|---|---|
| 목적 | 프로덕션 | A/B 비교 실험용 |
| 분할 기준 | 헤딩 트리 + Text/Table 분리 | 800자 윈도우 슬라이딩 |
| chunk_type | `text` / `table` / `image` | 미사용 |
| part_index/part_total | O (sibling 복원용) | 미사용 |
| 임베딩 텍스트 | `heading_path + content` | `content`만 |

## 크기 파라미터

BGE-M3 sweet spot: 256~512 tokens ≈ 800~1600 chars (한국어 기준)

| 설정 | 값 | 근거 |
|------|-----|------|
| TEXT_MAX_CHARS | 1200자 | sweet spot 중앙 (~400 tokens) |
| TABLE_MAX_CHARS | 2400자 | 마크다운 오버헤드 감안 시 실질 ~400 tokens |
| CHUNK_MIN_CHARS | 300자 | 자투리 병합 기준 |
| FIXED_WINDOW_SIZE | 800자 | sweet spot 하단 |
| FIXED_OVERLAP_SIZE | 150자 | 경계 문맥 보존 |

경험적 설정값. 실제 검색 품질을 보면서 조정.

## Adaptive 상세

### Text / Table / Iqmage 분리

| | 텍스트 | 테이블 | 이미지 OCR |
|---|---|---|---|
| 분할 기준 | TEXT_MAX_CHARS 초과 시 문장 경계 분할 | TABLE_MAX_CHARS 초과 시 행 분할 | OCR 결과 전체를 하나의 청크 |
| 자투리 병합 | CHUNK_MIN_CHARS 미만 → 이전 청크에 병합 | — | — |
| 행 분할 시 | — | 헤더(컬럼명 + 구분선) 매 청크 반복 | — |
| chunk_type | `text` | `table` | `image` |

### chunk_type 설계 원칙

**출처가 아니라 내용의 성격이 기준**이다. ODL markdown에서 파싱된 표든 paddle OCR로 이미지에서 복원한 표든, 행·열 구조가 있으면 `chunk_type="table"`. 평문 텍스트는 `chunk_type="text"`(markdown 본문) 또는 `chunk_type="image"`(이미지 OCR). LLM과 검색 레이어는 출처 구분 없이 동일하게 취급한다.

**분리 저장 + 메타데이터 연결**: 같은 이미지에서 나온 image/table 청크는 동일한 `heading_path`를 공유한다. `expand_siblings()` ([src/v1/rag/sibling.py](../src/v1/rag/sibling.py))가 heading_path 기준으로 청크를 묶어 가져오므로, 검색에서 table 청크가 hit되면 같은 조항의 image 청크(주변 설명)가 함께 LLM context에 들어간다. 합쳐서 한 청크로 만들면 표 구조가 평문에 섞여 LLM이 표로 인식 못 하므로 비표준.

### 이미지 OCR 파이프라인 (extract → ocr → chunk)

```
[extract] ODL이 PDF를 markdown으로 변환 + 내부 이미지 객체를 파일로 떨굼
          (image_output="external": 1px spacer, 투명 오버레이 등 garbage 포함)
          ↓ celery extract.py 후처리 (_prune_garbage_images)
          is_valid_image 6단계 필터로 garbage PNG 목록 선별 → ODL /cleanup API로 일괄 삭제
          (UID 1000 소유라 워커가 직접 rm 불가 → ODL 컨테이너의 cleanup API 경유)

[ocr]     celery가 markdown의 ![image N](...) 태그 파싱
          │
          ├── [1] is_valid_image 6단 입구 필터 (ODL object-level garbage filter)
          │   · 파일 크기     file_size < 100B
          │   · 최소 차원      dimension  < 10px
          │   · figure 최소    short side < 300x200 (아이콘·로고·QR)
          │   · 종횡비 상한    aspect     ≥ 10:1 (가로띠)
          │   · 최대 차원      dimension  > 7000px (대형 배너)
          │   · 단색 검출      stddev     < 5 (단색·투명 마스크)
          │   — garbage 이미지는 여기서 컷, paddle 호출 안 함
          │
          ├── [2] paddle HTTP 호출 → PP-StructureV3 처리 (개별 이미지 파일만)
          │
          ├── [3] _extract_blocks 라벨 분류
          │   drop: header/footer/page_number/... → 청크 제외
          │   table: → chunk_type="table" 개별 청크
          │   나머지: text/title/caption/formula/... → chunk_type="image" 합친 청크
          │
          └── [4] is_meaningful_ocr_result + heading 중복(Jaccard>0.7) → 최종 드롭

[chunk]   markdown text 청크 + OCR 청크(image + table) 합류 → part_index 재부여
```

`paddle`은 이미지 파일만 처리하며 PDF를 직접 다루지 않는다.

### 필터 임계값 (config/settings.py)

| 상수 | 값 | 역할 |
|---|---|---|
| `OCR_MIN_FILE_SIZE` | 100 | 파일 크기 필터 — 빈 PNG 차단 |
| `OCR_MIN_IMAGE_WIDTH/HEIGHT` | 10 | 최소 차원 필터 — 1~2px spacer 차단 |
| `OCR_FIGURE_MIN_WIDTH/HEIGHT` | 300/200 | figure 최소 크기 필터 — 아이콘·로고·QR 차단 |
| `OCR_MAX_ASPECT_RATIO` | 10 | 종횡비 필터 — 페이지 구분선·가로띠 차단 |
| `OCR_MAX_IMAGE_DIMENSION` | 7000 | 최대 차원 필터 — 대형 배너 차단 (리사이즈 오버헤드 방지) |
| `OCR_MIN_PIXEL_STDDEV` | 5.0 | 단색 필터 — 빈 배경·투명 마스크 차단 |
| `OCR_MIN_TEXT_LENGTH` | 10 | 출구 필터 — 너무 짧은 OCR 결과 차단 |

### OCR 통계 로그

```
[OCR 통계] {doc} — 전체 N, 입구필터 M [file_size:x, icon_size:x, flat_color:x, ...],
           빈값 x, drop전용 x, 노이즈 x, 중복 x, 저장(image) x, 저장(table) x
```

- `drop전용`: header/footer만 있어서 콘텐츠 블록 0인 이미지
- `저장(image)`: 콘텐츠 텍스트 청크로 저장된 건수
- `저장(table)`: 표 청크 건수 (한 이미지에 여러 표 가능)

### 엔진 설정

`PPStructureV3(lang="korean", device="cpu", enable_mkldnn=False)` — CPU 모드 고정. Blackwell sm_120이 현재 `paddlepaddle/paddle:3.3.1-gpu-cuda13.0-cudnn9.13` 빌드에 미포함이라 GPU 초기화 불가. PIR+oneDNN 경로의 `NotImplementedError` 우회를 위해 `enable_mkldnn=False` 생성자 + `FLAGS_use_mkldnn=0` / `FLAGS_enable_pir_in_executor=0` env 3중 차단. 자세한 배경은 [architecture.md](architecture.md).

### 저장 산출물

각 이미지 옆에 `_ocr.json` + `_ocr_layout.png` 저장.

`_ocr.json` 스키마 (PP-StructureV3 결과):
```json
{
  "input_path": "...",
  "width": 1483, "height": 1082,
  "rec_texts": ["..."],           // REC_MIN_SCORE=0.5 이상만
  "rec_scores": [0.95, ...],      // REC_MIN_SCORE=0.5 이상만
  "layout_boxes": [{"label": "text", "score": 0.9, "bbox": [x1,y1,x2,y2]}],
  "parsing_blocks": [{"label": "...", "content": "..."}]
}
```

**저장 정책**: `is_valid_image` 입구 필터(1계층)를 통과해 paddle로 넘어온 이미지의 **파싱 결과 구조를 왜곡 없이** 저장하되, 저장 시점에 confidence 컷을 걸어 garbage 검출은 배제한다:

- **전처리 없음**: 이진화·블러 등 input 전처리 금지. PP-StructureV3 layout detector는 자연 RGB로 학습돼 이진화하면 layout_boxes가 10배 줄고 라벨이 뭉개짐 (실측 확인, 1 vs 10).
- **저장 컷**: `LAYOUT_MIN_SCORE=0.5` 이상인 layout_boxes만 JSON/PNG에 기록, `REC_MIN_SCORE=0.5` 이상인 rec_texts만 기록. 두 컷 모두 통과하는 게 하나도 없으면 `_ocr.json`/`_ocr_layout.png` 생성 자체를 스킵 (garbage 이미지로 디스크 낭비 방지).
- **의미 필터 위치**: drop/table/text 라벨 분류, heading 중복, 한글비율 체크는 전부 청킹 단계(3계층)에서. paddle 단계는 confidence 기반 물리적 컷만 담당.

`_ocr_layout.png`는 기본적으로 레이아웃 블록만 색상별 박스로 그리며, 환경변수 `PADDLE_VIZ_TEXT_LINES=1`이면 개별 텍스트 라인을 자홍색으로 추가 오버레이 (디버깅용).

필터 효과 검증: `scripts/eval_ocr.py`.

### Sibling 복원 (검색 시점)

같은 `heading_path` 내 `part_index`로 순서 추적:

```
예: "제4조" heading_path 내
  text   (part 1/4)  ← 검색 hit
  image  (part 2/4)  ← sibling 복원으로 같이 가져옴
  table  (part 3/4)  ← 같이 가져옴
  text   (part 4/4)  ← 같이 가져옴
```

hit된 part_index 기준 ±`SIBLING_WINDOW`(기본 2)개만 가져온다. 전체가 아닌 슬라이딩 윈도우. "검색은 작게, LLM 전달은 크게". chunk_type 무관, heading_path만으로 묶이므로 text/image/table이 자연스럽게 같은 섹션으로 복원된다.

### 이미지 OCR 청크 크기 리스크

BGE-M3는 8k 토큰까지 지원하지만, 512~1024 토큰 내에서 retrieval 품질이 가장 안정적이라는 보고가 있다. OCR 텍스트가 sweet spot(800~1600자)을 초과하면 임베딩 품질 저하 가능. 레이아웃 보존을 위해 분할하지 않는 보수적 전략이지만, 운영 데이터에서 이미지 청크 길이 분포를 주기적으로 확인하고 초과 비율이 높으면 bbox 기반 서브블록 분리 검토.

## Fixed 상세

- 800자 윈도우 + 150자 오버랩, 단어 중간 절단 방지
- TOC 제거 후 구조 무시하고 기계적 분할
- 각 윈도우에 가장 가까운 이전 헤딩을 메타데이터로 태깅 (content에는 미포함)

## JSON 출력 예시

### Adaptive — 텍스트
```json
{
  "service_code": "01",
  "source_file": "약관.pdf",
  "page_start": 15, "page_end": 15,
  "heading_path": ["제1장 일반사항", "제4조 보험금의 지급사유"],
  "heading": "제4조 보험금의 지급사유",
  "chunk_type": "text",
  "part_index": 1, "part_total": 4,
  "char_count": 142,
  "content": "회사는 보험기간 중 피보험자에게..."
}
```

### Adaptive — 이미지 OCR (콘텐츠 블록 합친 것)
```json
{
  "chunk_type": "image",
  "part_index": 2, "part_total": 4,
  "heading_path": ["제1장 일반사항", "제4조 보험금의 지급사유"],
  "image_paths": ["약관_images/img8.png"],
  "image_ocr_texts": ["보험금 지급사유\n1. 사망\n2. 후유장해\n..."],
  "char_count": 35,
  "content": "보험금 지급사유\n1. 사망\n2. 후유장해\n..."
}
```

### Adaptive — 테이블 (같은 이미지에서 분리됨)
```json
{
  "chunk_type": "table",
  "part_index": 3, "part_total": 4,
  "heading_path": ["제1장 일반사항", "제4조 보험금의 지급사유"],
  "image_paths": ["약관_images/img8.png"],
  "image_ocr_texts": ["| 구분 | 지급률 |\n| --- | --- |\n| 사망 | 100% |"],
  "char_count": 48,
  "content": "| 구분 | 지급률 |\n| --- | --- |\n| 사망 | 100% |"
}
```

> 같은 `image_paths`를 공유하는 image 청크와 table 청크는 동일한 `heading_path`로 묶여 sibling 복원에서 함께 LLM context에 들어간다. PP-StructureV3의 `table_res_list` HTML은 celery의 `_html_table_to_markdown()`이 마크다운 테이블로 변환.

### Fixed
```json
{
  "service_code": "01",
  "source_file": "약관.pdf",
  "page_start": 15, "page_end": 16,
  "heading_path": ["제1장 일반사항", "제4조 보험금의 지급사유"],
  "heading": "제4조 보험금의 지급사유",
  "char_count": 796,
  "content": "회사는 보험기간 중 피보험자에게..."
}
```

Fixed는 `chunk_type`, `part_index`, `part_total`, `image_paths` 없음.

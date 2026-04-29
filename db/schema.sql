-- ============================================================
-- docs-rag 테이블 DDL
-- 실행 전 기존 테이블 백업 필수.
-- ============================================================

-- 기존 테이블 제거 (의존성 역순)
DROP TABLE IF EXISTS tb_query_feedback;
DROP TABLE IF EXISTS tb_document_contents;
DROP TABLE IF EXISTS tb_document_chunks;
DROP TABLE IF EXISTS tb_document_extract;
DROP TABLE IF EXISTS tb_document_status;
DROP TABLE IF EXISTS tb_document_status_log;
DROP TABLE IF EXISTS tb_code_master;
DROP TABLE IF EXISTS tb_service_code;


-- ============================================================
-- 1. 서비스 코드 마스터
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_service_code (
    service_code VARCHAR(10) PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL
);

COMMENT ON TABLE tb_service_code IS '서비스 코드 마스터';
COMMENT ON COLUMN tb_service_code.service_code IS '서비스 구분 코드 (2자리, PK). 예: 01=AI_PARSER, 02=AIKON';
COMMENT ON COLUMN tb_service_code.service_name IS '서비스 한글/영문 이름';


-- ============================================================
-- 2. 상태 코드 마스터
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_code_master (
    code      VARCHAR(10) PRIMARY KEY,
    code_name VARCHAR(100) NOT NULL
);

COMMENT ON TABLE tb_code_master IS '상태 코드 마스터 (파이프라인 status_code 룩업 테이블)';
COMMENT ON COLUMN tb_code_master.code IS '상태 코드 값 (2자리, PK). 00=대기, 11=완료, 21~43=단계별 처리/완료, 91~99=에러';
COMMENT ON COLUMN tb_code_master.code_name IS '상태 한글 설명 (예: "완료(임베딩)")';


-- ============================================================
-- 3. 상태 변경 이력 (CQRS write model, append-only)
--    파이프라인 상태 변경 시 INSERT만 — 이 테이블이 상태의 source of truth.
--    단계별 소요시간 계산·디버깅·감사 추적에 사용.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_document_status_log (
    id            BIGSERIAL PRIMARY KEY,
    service_code  VARCHAR(2) NOT NULL,
    document_id   VARCHAR(255) NOT NULL,
    from_status   VARCHAR(2),
    to_status     VARCHAR(2) NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_status_log_lookup ON tb_document_status_log(service_code, document_id);
CREATE INDEX idx_status_log_created ON tb_document_status_log(created_at);

COMMENT ON TABLE tb_document_status_log IS '상태 변경 이력 — append-only 원본 (CQRS write model)';
COMMENT ON COLUMN tb_document_status_log.id IS 'PK (BIGSERIAL)';
COMMENT ON COLUMN tb_document_status_log.service_code IS '서비스 구분 코드 (tb_service_code FK 미설정 — 성능·운영 단순화)';
COMMENT ON COLUMN tb_document_status_log.document_id IS '문서 식별자 (서비스 내 unique)';
COMMENT ON COLUMN tb_document_status_log.from_status IS '변경 전 상태 코드 (최초 등록 시 NULL)';
COMMENT ON COLUMN tb_document_status_log.to_status IS '변경 후 상태 코드';
COMMENT ON COLUMN tb_document_status_log.created_at IS '상태 변경 시각 (단계별 소요시간 계산 기준)';


-- ============================================================
-- 4. 현재 상태 스냅샷 (CQRS read model, 파생)
--    tb_document_status_log에서 파생된 현재 상태. 빠른 조회용.
--    UPDATE로 최신 상태만 유지. 문서 단위 완료 = status_code '11'.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_document_status (
    id            BIGSERIAL PRIMARY KEY,
    service_code  VARCHAR(2) NOT NULL,
    document_id   VARCHAR(255) NOT NULL,
    document_name VARCHAR(255) NOT NULL,
    document_path VARCHAR(500),
    status_code   VARCHAR(2) NOT NULL DEFAULT '00',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP,
    UNIQUE (service_code, document_id)
);

CREATE INDEX idx_status_code ON tb_document_status(status_code);

COMMENT ON TABLE tb_document_status IS '현재 상태 스냅샷 (CQRS read model). 원본은 tb_document_status_log';
COMMENT ON COLUMN tb_document_status.id IS 'PK (BIGSERIAL)';
COMMENT ON COLUMN tb_document_status.service_code IS '서비스 구분 코드 (UNIQUE 제약의 일부)';
COMMENT ON COLUMN tb_document_status.document_id IS '문서 식별자 (UNIQUE 제약의 일부 — 같은 service_code 내 중복 불가)';
COMMENT ON COLUMN tb_document_status.document_name IS '원본 PDF 파일명 (확장자 포함)';
COMMENT ON COLUMN tb_document_status.document_path IS '원본 파일 경로 (메타데이터용, 실제 추출은 ODL 컨테이너가 처리)';
COMMENT ON COLUMN tb_document_status.status_code IS '파이프라인 현재 상태 코드 (default 00=대기, 11=전체완료)';
COMMENT ON COLUMN tb_document_status.created_at IS '문서 등록 시각';
COMMENT ON COLUMN tb_document_status.updated_at IS '마지막 상태 변경 시각 (update_status() 호출 시 갱신)';


-- ============================================================
-- 5. 추출 원본 보존
--    문서 추출 결과(JSON + Markdown) 보존. 재청킹 시 raw_markdown 재사용.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_document_extract (
    id            BIGSERIAL PRIMARY KEY,
    service_code  VARCHAR(2) NOT NULL,
    document_id   VARCHAR(255) NOT NULL,
    document_name VARCHAR(255),
    document_path VARCHAR(500),
    total_pages   INT,
    raw_json      JSONB,
    raw_markdown  TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_extract_lookup ON tb_document_extract(service_code, document_id);

COMMENT ON TABLE tb_document_extract IS '문서 추출 원본 보존 (재청킹 시 원본 소스)';
COMMENT ON COLUMN tb_document_extract.id IS 'PK (BIGSERIAL)';
COMMENT ON COLUMN tb_document_extract.service_code IS '서비스 구분 코드';
COMMENT ON COLUMN tb_document_extract.document_id IS '문서 식별자';
COMMENT ON COLUMN tb_document_extract.document_name IS 'PDF 파일명 (status 테이블과 동일 — denormalize, 디버깅 편의)';
COMMENT ON COLUMN tb_document_extract.document_path IS '원본 파일 경로';
COMMENT ON COLUMN tb_document_extract.total_pages IS 'PDF 총 페이지 수 (DOCLING_PAGE_LIMIT 분기 판단용)';
COMMENT ON COLUMN tb_document_extract.raw_json IS '추출 결과 JSON 원본 (ODL 응답 그대로)';
COMMENT ON COLUMN tb_document_extract.raw_markdown IS '추출 결과 마크다운 원본 (재청킹 시 chunker 입력)';
COMMENT ON COLUMN tb_document_extract.created_at IS '추출 완료 시각';


-- ============================================================
-- 6. 청킹 결과
--    마크다운 헤딩 트리 기반 청킹 결과 보존. 재임베딩 시 content 재사용.
--    임베딩 진행률 = LEFT JOIN tb_document_contents → 없는 row가 미처리 청크.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    service_code    VARCHAR(2) NOT NULL,
    document_id     VARCHAR(255) NOT NULL,
    seq             INT NOT NULL,
    heading         TEXT,
    heading_path    TEXT,
    content         TEXT NOT NULL,
    char_count      INT,
    start_page      INT,
    end_page        INT,
    chunk_type      VARCHAR(16),
    chunk_strategy  VARCHAR(16),
    part_index      INT,
    part_total      INT,
    image_paths     JSONB,
    image_ocr_texts JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_lookup ON tb_document_chunks(service_code, document_id);
CREATE INDEX idx_chunks_heading ON tb_document_chunks(heading_path);

COMMENT ON TABLE tb_document_chunks IS '문서 청크 데이터 (청킹 결과 보존)';
COMMENT ON COLUMN tb_document_chunks.id IS 'PK (BIGSERIAL) — tb_document_contents.chunk_id가 참조';
COMMENT ON COLUMN tb_document_chunks.service_code IS '서비스 구분 코드';
COMMENT ON COLUMN tb_document_chunks.document_id IS '문서 식별자';
COMMENT ON COLUMN tb_document_chunks.seq IS '청크 순서 (1부터, 문서 내 글로벌 순번)';
COMMENT ON COLUMN tb_document_chunks.heading IS '청크가 속한 가장 가까운 헤딩 (마지막 노드)';
COMMENT ON COLUMN tb_document_chunks.heading_path IS '헤딩 경로 (예: 제1장 > 제1조). sibling 복원·임베딩 텍스트에 사용';
COMMENT ON COLUMN tb_document_chunks.content IS '청크 본문 — heading_path는 임베딩/LLM 시점에 합침';
COMMENT ON COLUMN tb_document_chunks.char_count IS '본문 글자 수 (CHUNK_MIN_CHARS·MAX_CHARS 판정 기준)';
COMMENT ON COLUMN tb_document_chunks.start_page IS '청크가 시작되는 페이지 번호';
COMMENT ON COLUMN tb_document_chunks.end_page IS '청크가 끝나는 페이지 번호 (한 청크가 여러 페이지 걸치면 다름)';
COMMENT ON COLUMN tb_document_chunks.chunk_type IS 'text / table / image — 출처가 아닌 내용 성격 기준 (chunking.md 참조)';
COMMENT ON COLUMN tb_document_chunks.chunk_strategy IS 'adaptive (헤딩 기반, 프로덕션) / fixed (윈도우 슬라이딩, A/B 비교용)';
COMMENT ON COLUMN tb_document_chunks.part_index IS '같은 heading_path 내 분할 순번 (sibling 복원용, 1-based)';
COMMENT ON COLUMN tb_document_chunks.part_total IS '같은 heading_path 내 분할 총 수';
COMMENT ON COLUMN tb_document_chunks.image_paths IS '이미지/표 청크의 원본 이미지 파일 경로 배열 (JSONB list[str])';
COMMENT ON COLUMN tb_document_chunks.image_ocr_texts IS '이미지/표 청크의 PaddleOCR 추출 텍스트 배열 (JSONB list[str])';
COMMENT ON COLUMN tb_document_chunks.created_at IS '청킹 완료 시각';


-- ============================================================
-- 7. 검색 서빙 테이블
--    임베딩 적재: chunks 기반 벡터 생성 → 벡터DB 저장 → 여기에 INSERT.
--    검색 서빙: 벡터DB 검색 → qdrant_point_id로 단건 조회 → JOIN 없이 응답.
--    문서 단위 완료 판정은 tb_document_status.status_code = '11'.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_document_contents (
    id               BIGSERIAL PRIMARY KEY,
    service_code     VARCHAR(2) NOT NULL,
    document_id      VARCHAR(255) NOT NULL,
    chunk_id         BIGINT,

    heading          TEXT,
    heading_path     TEXT,
    content          TEXT,

    start_page       INT,
    end_page         INT,
    chunk_type       VARCHAR(16),
    chunk_strategy   VARCHAR(16),
    part_index       INT,
    part_total       INT,
    image_paths      JSONB,
    image_ocr_texts  JSONB,

    qdrant_point_id  BIGINT,
    token_count      INT,
    char_count       INT,

    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_contents_lookup ON tb_document_contents(service_code, document_id);
CREATE INDEX idx_contents_qdrant ON tb_document_contents(qdrant_point_id);
CREATE INDEX idx_contents_page ON tb_document_contents(start_page, end_page);

COMMENT ON TABLE tb_document_contents IS '검색 서빙 테이블 (임베딩 적재 + 검색 응답)';
COMMENT ON COLUMN tb_document_contents.id IS 'PK (BIGSERIAL)';
COMMENT ON COLUMN tb_document_contents.service_code IS '서비스 구분 코드';
COMMENT ON COLUMN tb_document_contents.document_id IS '문서 식별자';
COMMENT ON COLUMN tb_document_contents.chunk_id IS 'tb_document_chunks.id 참조 (FK 미설정 — 운영 단순화)';
COMMENT ON COLUMN tb_document_contents.heading IS '가장 가까운 헤딩 (chunks 테이블에서 복사)';
COMMENT ON COLUMN tb_document_contents.heading_path IS '헤딩 경로 (sibling 복원 키)';
COMMENT ON COLUMN tb_document_contents.content IS '청크 본문 (검색 응답에 그대로 노출)';
COMMENT ON COLUMN tb_document_contents.start_page IS '청크 시작 페이지';
COMMENT ON COLUMN tb_document_contents.end_page IS '청크 끝 페이지';
COMMENT ON COLUMN tb_document_contents.chunk_type IS 'text / table / image';
COMMENT ON COLUMN tb_document_contents.chunk_strategy IS 'adaptive / fixed';
COMMENT ON COLUMN tb_document_contents.part_index IS '같은 heading_path 내 분할 순번';
COMMENT ON COLUMN tb_document_contents.part_total IS '같은 heading_path 내 분할 총 수';
COMMENT ON COLUMN tb_document_contents.image_paths IS '이미지/표 청크의 원본 이미지 경로 배열 (JSONB)';
COMMENT ON COLUMN tb_document_contents.image_ocr_texts IS '이미지/표 청크의 OCR 추출 텍스트 배열 (JSONB)';
COMMENT ON COLUMN tb_document_contents.qdrant_point_id IS 'Qdrant 벡터DB 포인트 ID (BIGINT — Qdrant가 unsigned integer 또는 UUID만 허용. chunks.id와 동일 값)';
COMMENT ON COLUMN tb_document_contents.token_count IS '임베딩 모델(BGE-M3) 기준 토큰 수';
COMMENT ON COLUMN tb_document_contents.char_count IS '본문 글자 수';
COMMENT ON COLUMN tb_document_contents.created_at IS '서빙 테이블 적재 시각';


-- ============================================================
-- 8. 쿼리 피드백 (Insert-only)
--    /answer·/retrieve 응답의 trace_id를 받아 사용자 피드백 수집.
--    외래키 없음 — trace는 JSONL 파일이라 DB FK 불가 + race condition 방지.
--    집계는 scripts/feedback_summary.py가 trace_id로 JSONL과 조인.
-- ============================================================
CREATE TABLE IF NOT EXISTS tb_query_feedback (
    id          BIGSERIAL PRIMARY KEY,
    trace_id    VARCHAR(64) NOT NULL,
    signal      VARCHAR(20) NOT NULL,
    free_text   TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_feedback_signal CHECK (signal IN ('up', 'down', 'reformulated'))
);

CREATE INDEX idx_feedback_trace_id ON tb_query_feedback(trace_id);
CREATE INDEX idx_feedback_created ON tb_query_feedback(created_at);

COMMENT ON TABLE tb_query_feedback IS '쿼리별 사용자 피드백 (Insert-only). trace_id로 서빙 trace JSONL과 조인';
COMMENT ON COLUMN tb_query_feedback.id IS 'PK (BIGSERIAL)';
COMMENT ON COLUMN tb_query_feedback.trace_id IS 'TraceRecord.trace_id (UUID v4 문자열, FK 없음 — JSONL과 조인 키)';
COMMENT ON COLUMN tb_query_feedback.signal IS 'up(좋음) / down(나쁨) / reformulated(재질문) — CHECK 제약 강제';
COMMENT ON COLUMN tb_query_feedback.free_text IS '사용자 자유 서술 (선택, max 2000자 — API 레이어 검증)';
COMMENT ON COLUMN tb_query_feedback.created_at IS '피드백 수신 시각';


-- ============================================================
-- 초기 데이터
-- ============================================================

INSERT INTO tb_service_code (service_code, service_name) VALUES
    ('01', 'AI_PARSER'),
    ('02', 'AIKON'),
    ('03', 'OMNISIGNAL')
ON CONFLICT (service_code) DO NOTHING;

INSERT INTO tb_code_master (code, code_name) VALUES
    ('00', '대기'),
    ('11', '완료(전체)'),
    ('21', '완료(PDF추출)'),
    ('22', '처리중(PDF추출)'),
    ('23', '완료(OCR)'),
    ('24', '처리중(OCR)'),
    ('31', '완료(청킹/DB적재)'),
    ('32', '처리중(청킹)'),
    ('41', '완료(벡터DB 적재)'),
    ('42', '처리중(임베딩)'),
    ('43', '완료(임베딩)'),
    ('91', '에러(PDF추출)'),
    ('92', '에러(OCR)'),
    ('93', '에러(청킹)'),
    ('94', '에러(청킹/DB적재)'),
    ('95', '에러(임베딩)'),
    ('96', '에러(임베딩/벡터DB적재)'),
    ('99', '에러(기타)')
ON CONFLICT (code) DO NOTHING;

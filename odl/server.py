"""ODL HTTP API.

Why this wrapper exists (중요):
    celery worker가 PDF 변환하려면 opendataloader-pdf를 호출해야 하는데,
    예전엔 `docker exec odl ...` 방식이었음. 이러면 celery 컨테이너 안에
    /var/run/docker.sock을 마운트해야 하고, 그 순간 celery는 호스트 도커 데몬
    전체 제어권을 가짐(다른 컨테이너 kill, privileged 컨테이너 생성 등) —
    컨테이너 격리가 사실상 깨지고, Kubernetes로 이식도 불가.

    그래서 이 파일이 존재: opendataloader_pdf.convert()를 HTTP로 감싸서
    celery가 compose 내부 네트워크로 깔끔하게 호출하게 함. docker.sock 제거가
    원래 동기. 이후 /convert 위에 fallback 분기와 /cleanup(UID 1000 파일 삭제)이
    얹혔지만, 본질은 컨테이너 경계를 깨끗이 하는 것.
"""

import shutil
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import opendataloader_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("odl-server")

app = FastAPI(title="ODL Convert API", version="0.1.0")


class ConvertRequest(BaseModel):
    input_path: str
    output_dir: str
    format: str = "json,markdown"
    image_output: str = "external"
    image_format: str = "png"
    markdown_page_separator: str = "<!-- page:%page-number% -->"
    hybrid: Optional[str] = None
    hybrid_mode: str = "full"
    hybrid_url: Optional[str] = None
    hybrid_timeout: Optional[str] = None


class CleanupRequest(BaseModel):
    paths: list[str]


@app.post("/convert")
def convert(req: ConvertRequest):
    """PDF → Markdown/JSON 변환. opendataloader_pdf.convert() 래퍼."""
    logger.info(f"[convert] {req.input_path} (hybrid={req.hybrid})")

    kwargs = {
        "input_path": req.input_path,
        "output_dir": req.output_dir,
        "format": req.format,
        "image_output": req.image_output,
        "image_format": req.image_format,
        "markdown_page_separator": req.markdown_page_separator,
    }
    if req.hybrid:
        kwargs["hybrid"] = req.hybrid
        kwargs["hybrid_mode"] = req.hybrid_mode
        if req.hybrid_url:
            kwargs["hybrid_url"] = req.hybrid_url
        if req.hybrid_timeout:
            kwargs["hybrid_timeout"] = req.hybrid_timeout

    try:
        opendataloader_pdf.convert(**kwargs)
        return {"status": "ok"}
    except Exception as e:
        error_id = str(uuid4())[:8]
        logger.error(f"[convert] 실패 [{error_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"변환 실패. error_id: {error_id}")


_CLEANUP_BASE = Path("/data")  # 이 디렉토리 하위만 삭제 허용


@app.post("/cleanup")
def cleanup(req: CleanupRequest):
    """ODL 소유 파일 삭제. Worker에서 직접 삭제 불가한 파일(UID 불일치) 대응.
    경로 통과(path traversal) 방지: /data/ 하위만 허용."""
    removed = []
    for p in req.paths:
        path = Path(p).resolve()
        if not str(path).startswith(str(_CLEANUP_BASE.resolve())):
            logger.warning(f"[cleanup] 허용 범위 밖 경로 차단: {p}")
            continue
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(p)
    return {"removed": removed}


@app.get("/health")
def health():
    return {"status": "ok"}

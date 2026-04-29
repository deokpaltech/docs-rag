"""로깅 설정 — 파일 + stdout 동시 출력.

같은 메시지를 LOG_DIR/{api,celery}.log 파일과 stdout 양쪽에 송출 → docker compose
logs와 host 파일 모두에서 추적 가능. handlers 중복 등록 방지를 위해 setup 시
체크 (모듈 reimport 시 같은 핸들러가 N번 붙어 로그가 N배 찍히는 회귀 차단).
로테이션 미적용 — 장기 운영 시 RotatingFileHandler 교체 필요.
"""

import logging
from .config.settings import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(sh)

    return logger


api_logger = _setup_logger("api", "api.log")
celery_logger = _setup_logger("celery", "celery.log")

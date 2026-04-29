"""SQLAlchemy 엔진 + 세션 팩토리.

pool_pre_ping=True: 유휴 커넥션이 RDS·docker network에서 끊긴 경우 재연결
(첫 쿼리 OperationalError 회귀 차단). FastAPI 의존성 주입은 get_db,
Celery task는 task_session — 후자는 except 시 자동 rollback이라 task 레벨
부분 실패가 다음 task에 dirty 세션을 남기지 않음.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

PG_CONFIG = {
    "host": os.environ["PG_HOST"],
    "dbname": os.environ["PG_DBNAME"],
    "user": os.environ["PG_USER"],
    "password": os.environ["PG_PASSWORD"],
    "port": int(os.environ["PG_PORT"]),
}

DATABASE_URL = URL.create(
    drivername="postgresql",
    username=PG_CONFIG["user"],
    password=PG_CONFIG["password"],
    host=PG_CONFIG["host"],
    port=PG_CONFIG["port"],
    database=PG_CONFIG["dbname"],
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends용 세션 제공"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def task_session() -> Generator[Session, None, None]:
    """Celery 태스크용 세션 (자동 rollback/close)"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

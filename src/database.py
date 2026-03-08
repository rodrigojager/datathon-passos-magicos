from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL


Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {}
        if DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            DATABASE_URL,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    import src.db_models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def record_event(level: str, logger_name: str, message: str, context: dict[str, object] | None = None) -> None:
    from src.db_models import AppEvent
    from src.utils import sanitize_record

    try:
        with session_scope() as session:
            session.add(
                AppEvent(
                    level=level,
                    logger=logger_name,
                    message=message,
                    context_json=sanitize_record(context or {}),
                )
            )
    except Exception:
        pass

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from . import models  # noqa: F401  # Ensure model tables are registered
from .models import Base, SCHEMA_NAMES

POSTGRES_DSN_ENV = "POSTGRES_DSN"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    project_root = Path(__file__).resolve().parent.parent
    fallback_env = project_root / ".ai" / ".env"
    if fallback_env.exists():
        load_dotenv(fallback_env, override=False)


def get_postgres_dsn() -> str:
    _load_dotenv_if_available()
    dsn = os.getenv(POSTGRES_DSN_ENV, "").strip()
    if not dsn:
        raise RuntimeError(
            f"Missing required environment variable: {POSTGRES_DSN_ENV}"
        )
    return dsn


def _normalize_postgres_dsn_for_sqlalchemy(dsn: str) -> str:
    value = str(dsn or "").strip()
    if value.startswith("postgresql+psycopg://"):
        return value
    if value.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + value[len("postgresql+psycopg2://") :]
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://") :]
    return value


def create_db_engine(*, dsn: str | None = None, echo: bool = False) -> Engine:
    return sa.create_engine(
        _normalize_postgres_dsn_for_sqlalchemy(dsn or get_postgres_dsn()),
        echo=echo,
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def ensure_schemas(engine: Engine, schemas: Sequence[str] = SCHEMA_NAMES) -> None:
    with engine.begin() as conn:
        for schema_name in schemas:
            conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')


def initialize_database(*, echo: bool = False, dsn: str | None = None) -> Engine:
    engine = create_db_engine(dsn=dsn, echo=echo)
    ensure_schemas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@contextmanager
def session_scope(*, dsn: str | None = None, echo: bool = False) -> Iterator[Session]:
    engine = create_db_engine(dsn=dsn, echo=echo)
    SessionLocal = create_session_factory(engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

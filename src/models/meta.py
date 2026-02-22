from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class JobRun(Base):
    __tablename__ = "job_run"
    __table_args__ = (
        sa.Index("ix_meta_job_run_job_name", "job_name"),
        sa.Index("ix_meta_job_run_status", "status"),
        sa.Index("ix_meta_job_run_started_at", "started_at"),
        {"schema": "meta"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    job_name: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        sa.String(16),
        nullable=False,
        server_default=sa.text("'running'"),
    )
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    full_refresh: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        server_default=sa.text("FALSE"),
    )
    run_params: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    run_stats: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    error_message: Mapped[str | None] = mapped_column(sa.Text)


class Symbol(TimestampMixin, Base):
    __tablename__ = "symbols"
    __table_args__ = (
        sa.UniqueConstraint("ticker", name="uq_meta_symbols_ticker"),
        sa.Index("ix_meta_symbols_exchange", "exchange"),
        sa.Index("ix_meta_symbols_is_active", "is_active"),
        {"schema": "meta"},
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    name: Mapped[str | None] = mapped_column(sa.String(255))
    exchange: Mapped[str | None] = mapped_column(sa.String(64))
    market: Mapped[str | None] = mapped_column(sa.String(64))
    currency: Mapped[str | None] = mapped_column(sa.String(16))
    timezone: Mapped[str | None] = mapped_column(sa.String(64))
    sector: Mapped[str | None] = mapped_column(sa.String(128))
    industry: Mapped[str | None] = mapped_column(sa.String(128))
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        server_default=sa.text("TRUE"),
    )
    data_source: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'yfinance'"),
    )
    source_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )

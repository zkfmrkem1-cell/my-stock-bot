from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DailyOHLCV(Base):
    __tablename__ = "ohlcv_daily"
    __table_args__ = (
        sa.CheckConstraint("high >= low", name="high_gte_low"),
        sa.CheckConstraint("volume >= 0", name="volume_non_negative"),
        sa.Index("ix_raw_ohlcv_daily_trade_date", "trade_date"),
        sa.Index("ix_raw_ohlcv_daily_job_run_id", "job_run_id"),
        {"schema": "raw"},
    )

    symbol_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("meta.symbols.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    trade_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    job_run_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        sa.ForeignKey("meta.job_run.id", ondelete="SET NULL"),
    )
    open: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    high: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    low: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    close: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    volume: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    dividends: Mapped[float] = mapped_column(
        DOUBLE_PRECISION,
        nullable=False,
        server_default=sa.text("0"),
    )
    stock_splits: Mapped[float] = mapped_column(
        DOUBLE_PRECISION,
        nullable=False,
        server_default=sa.text("0"),
    )
    source: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'yfinance'"),
    )
    qc_status: Mapped[str] = mapped_column(
        sa.String(16),
        nullable=False,
        server_default=sa.text("'pending'"),
    )
    qc_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    ingested_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )

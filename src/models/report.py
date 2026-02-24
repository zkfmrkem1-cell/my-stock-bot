from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DailyReport(Base):
    __tablename__ = "daily_reports"
    __table_args__ = (
        sa.UniqueConstraint("report_date", "report_type", name="uq_report_daily_reports_date_type"),
        sa.Index("ix_report_daily_reports_created_at", "created_at"),
        {"schema": "report"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    report_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    report_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(sa.String(255))
    model_name: Mapped[str | None] = mapped_column(sa.String(128))
    summary_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    highlight_symbols: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'[]'::jsonb"),
    )
    metrics_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    report_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    discord_sent: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        server_default=sa.text("FALSE"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


class SymbolReport(Base):
    __tablename__ = "symbol_reports"
    __table_args__ = (
        sa.UniqueConstraint(
            "symbol_id",
            "report_date",
            "report_type",
            "prompt_version",
            name="uq_report_symbol_reports_symbol_date_type_prompt",
        ),
        sa.Index("ix_report_symbol_reports_symbol_id", "symbol_id"),
        sa.Index("ix_report_symbol_reports_report_date", "report_date"),
        sa.Index("ix_report_symbol_reports_report_type", "report_type"),
        sa.Index("ix_report_symbol_reports_created_at", "created_at"),
        {"schema": "report"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("meta.symbols.id", ondelete="RESTRICT"),
        nullable=False,
    )
    report_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    report_type: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        server_default=sa.text("'daily_snapshot'"),
    )
    status: Mapped[str] = mapped_column(
        sa.String(16),
        nullable=False,
        server_default=sa.text("'ready'"),
    )
    schema_version: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'v1'"),
    )
    prompt_version: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        server_default=sa.text("''"),
    )
    model_name: Mapped[str | None] = mapped_column(sa.String(128))
    summary_text: Mapped[str | None] = mapped_column(sa.Text)
    source_news_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'[]'::jsonb"),
    )
    feature_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    metrics_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    report_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DailyLabel(Base):
    __tablename__ = "daily_labels"
    __table_args__ = (
        sa.Index("ix_label_daily_labels_trade_date", "trade_date"),
        {"schema": "label"},
    )

    symbol_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("meta.symbols.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    trade_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    fwd_return_1d: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    fwd_return_5d: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    fwd_return_20d: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    target_up_5d: Mapped[bool | None] = mapped_column(sa.Boolean)
    target_up_20d: Mapped[bool | None] = mapped_column(sa.Boolean)
    target_rebound_after_oversold: Mapped[bool | None] = mapped_column(sa.Boolean)
    label_version: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'v1'"),
    )
    label_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


from __future__ import annotations

from datetime import date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DailyFeature(Base):
    __tablename__ = "daily_features"
    __table_args__ = (
        sa.Index("ix_feat_daily_features_trade_date", "trade_date"),
        {"schema": "feat"},
    )

    symbol_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("meta.symbols.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    trade_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    close_price: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    return_1d: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    return_5d: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    ma_5: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    ma_20: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    ma_60: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    dist_ma20: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    rsi_14: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    vol_ratio_20: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    feature_version: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'v1'"),
    )
    feature_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class StockNews(Base):
    __tablename__ = "stock_news"
    __table_args__ = (
        sa.Index("ix_news_stock_news_symbol_id", "symbol_id"),
        sa.Index("ix_news_stock_news_published_at", "published_at"),
        {"schema": "news"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    symbol_id: Mapped[int] = mapped_column(sa.BigInteger, sa.ForeignKey("meta.symbols.id"), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source: Mapped[str] = mapped_column(sa.String(32), nullable=False)  # 'yfinance' | 'google_rss'
    url: Mapped[str | None] = mapped_column(sa.Text)
    sentiment_flag: Mapped[str] = mapped_column(
        sa.String(16),
        nullable=False,
        server_default=sa.text("'normal'"),
    )  # 'normal' | 'caution' | 'exclude'
    ai_comment: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )


class TradeSignal(Base):
    __tablename__ = "trade_signals"
    __table_args__ = (
        sa.Index("ix_news_trade_signals_signal_date", "signal_date"),
        {"schema": "news"},
    )

    symbol_id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.ForeignKey("meta.symbols.id"), primary_key=True, nullable=False
    )
    signal_date: Mapped[date] = mapped_column(sa.Date, primary_key=True, nullable=False)
    is_tradeable: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from ..database import create_db_engine


@dataclass(slots=True)
class RawQCSummary:
    checked_rows: int = 0
    checked_symbols: int = 0
    invalid_rows: int = 0
    missing_sessions_total: int = 0
    missing_sessions_by_symbol: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked_rows": self.checked_rows,
            "checked_symbols": self.checked_symbols,
            "invalid_rows": self.invalid_rows,
            "missing_sessions_total": self.missing_sessions_total,
            "missing_sessions_by_symbol": self.missing_sessions_by_symbol,
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tickers(tickers: Sequence[str] | None) -> list[str]:
    if not tickers:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        for part in str(raw).split(","):
            ticker = part.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            out.append(ticker)
    return out


def _reflect_raw_table(conn: sa.Connection) -> sa.Table:
    return sa.Table("ohlcv_daily", sa.MetaData(), schema="raw", autoload_with=conn)


def _resolve_key_column(raw_table: sa.Table) -> sa.Column:
    if "symbol_id" in raw_table.c:
        return raw_table.c.symbol_id
    if "ticker" in raw_table.c:
        return raw_table.c.ticker
    raise RuntimeError("raw.ohlcv_daily must contain either symbol_id or ticker column for QC.")


def _build_scope_filters(
    raw_table: sa.Table,
    *,
    key_column: sa.Column,
    symbol_ids: Sequence[int] | None,
    tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> list[sa.ColumnElement[bool]]:
    filters: list[sa.ColumnElement[bool]] = []

    if key_column.name == "symbol_id" and symbol_ids:
        filters.append(key_column.in_(list(symbol_ids)))
    if key_column.name == "ticker":
        normalized = _normalize_tickers(tickers)
        if normalized:
            filters.append(key_column.in_(normalized))
    if start_date is not None:
        filters.append(raw_table.c.trade_date >= start_date)
    if end_date is not None:
        filters.append(raw_table.c.trade_date <= end_date)
    return filters


def _load_grouped_ranges(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    key_column: sa.Column,
    symbol_ids: Sequence[int] | None,
    tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> list[dict[str, Any]]:
    stmt = (
        sa.select(
            key_column.label("entity_key"),
            sa.func.min(raw_table.c.trade_date).label("min_trade_date"),
            sa.func.max(raw_table.c.trade_date).label("max_trade_date"),
            sa.func.count().label("row_count"),
        )
        .group_by(key_column)
        .order_by(key_column.asc())
    )
    filters = _build_scope_filters(
        raw_table,
        key_column=key_column,
        symbol_ids=symbol_ids,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if filters:
        stmt = stmt.where(*filters)
    return [dict(row._mapping) for row in conn.execute(stmt)]


def _load_actual_dates_by_key(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    key_column: sa.Column,
    symbol_ids: Sequence[int] | None,
    tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> dict[str, set[date]]:
    stmt = sa.select(key_column.label("entity_key"), raw_table.c.trade_date).order_by(
        key_column.asc(), raw_table.c.trade_date.asc()
    )
    filters = _build_scope_filters(
        raw_table,
        key_column=key_column,
        symbol_ids=symbol_ids,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if filters:
        stmt = stmt.where(*filters)

    by_key: dict[str, set[date]] = {}
    for row in conn.execute(stmt):
        by_key.setdefault(str(row.entity_key), set()).add(row.trade_date)
    return by_key


def _compute_missing_sessions(
    grouped_ranges: Sequence[dict[str, Any]],
    actual_dates_by_key: dict[str, set[date]],
    *,
    calendar_name: str,
) -> tuple[int, dict[str, list[str]]]:
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar(calendar_name)
    total_missing = 0
    missing_by_key: dict[str, list[str]] = {}

    for row in grouped_ranges:
        entity_key = str(row["entity_key"])
        min_trade_date = row["min_trade_date"]
        max_trade_date = row["max_trade_date"]
        if min_trade_date is None or max_trade_date is None:
            continue

        schedule = calendar.schedule(
            start_date=min_trade_date.isoformat(),
            end_date=max_trade_date.isoformat(),
        )
        expected_dates = {idx.date() for idx in schedule.index}
        actual_dates = actual_dates_by_key.get(entity_key, set())
        missing_dates = sorted(expected_dates - actual_dates)
        if not missing_dates:
            continue

        total_missing += len(missing_dates)
        missing_by_key[entity_key] = [d.isoformat() for d in missing_dates]

    return total_missing, missing_by_key


def _load_invalid_row_keys(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    key_column: sa.Column,
    symbol_ids: Sequence[int] | None,
    tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> list[tuple[Any, date]]:
    required_cols = ("open", "high", "low", "close", "volume")
    missing_required = [col for col in required_cols if col not in raw_table.c]
    if missing_required:
        raise RuntimeError(
            f"raw.ohlcv_daily missing required QC columns: {', '.join(sorted(missing_required))}"
        )

    invalid_predicate = sa.or_(
        raw_table.c.open.is_(None),
        raw_table.c.high.is_(None),
        raw_table.c.low.is_(None),
        raw_table.c.close.is_(None),
        raw_table.c.volume.is_(None),
        raw_table.c.volume < 0,
        raw_table.c.high < raw_table.c.low,
        raw_table.c.open < raw_table.c.low,
        raw_table.c.open > raw_table.c.high,
        raw_table.c.close < raw_table.c.low,
        raw_table.c.close > raw_table.c.high,
    )
    stmt = sa.select(key_column.label("entity_key"), raw_table.c.trade_date).where(invalid_predicate)
    filters = _build_scope_filters(
        raw_table,
        key_column=key_column,
        symbol_ids=symbol_ids,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if filters:
        stmt = stmt.where(*filters)
    return [(row.entity_key, row.trade_date) for row in conn.execute(stmt)]


def _set_qc_status_pass(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    key_column: sa.Column,
    symbol_ids: Sequence[int] | None,
    tickers: Sequence[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> int:
    if "qc_status" not in raw_table.c or "qc_meta" not in raw_table.c:
        return 0

    stmt = sa.update(raw_table).values(
        qc_status="pass",
        qc_meta={
            "checked_at": _utcnow_iso(),
            "checker": "pandas_market_calendars",
        },
    )
    filters = _build_scope_filters(
        raw_table,
        key_column=key_column,
        symbol_ids=symbol_ids,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if filters:
        stmt = stmt.where(*filters)
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def _mark_invalid_rows_fail(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    key_column: sa.Column,
    invalid_keys: Sequence[tuple[Any, date]],
) -> int:
    if not invalid_keys:
        return 0
    if "qc_status" not in raw_table.c or "qc_meta" not in raw_table.c:
        return 0

    stmt = (
        sa.update(raw_table)
        .where(key_column == sa.bindparam("pk_entity_key"))
        .where(raw_table.c.trade_date == sa.bindparam("pk_trade_date"))
        .values(
            qc_status="fail",
            qc_meta=sa.bindparam("qc_meta", type_=JSONB),
        )
    )
    payloads = [
        {
            "pk_entity_key": entity_key,
            "pk_trade_date": trade_date,
            "qc_meta": {
                "checked_at": _utcnow_iso(),
                "checker": "pandas_market_calendars",
                "reason": "ohlcv_range_or_null_violation",
            },
        }
        for entity_key, trade_date in invalid_keys
    ]
    result = conn.execute(stmt, payloads)
    return int(result.rowcount or 0)


def run_raw_ohlcv_qc(
    *,
    symbol_ids: Sequence[int] | None = None,
    tickers: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    calendar_name: str = "NYSE",
    echo: bool = False,
    engine: Engine | None = None,
) -> RawQCSummary:
    db_engine = engine or create_db_engine(echo=echo)
    summary = RawQCSummary()

    with db_engine.connect() as conn:
        raw_table = _reflect_raw_table(conn)
        key_column = _resolve_key_column(raw_table)

        grouped_ranges = _load_grouped_ranges(
            conn,
            raw_table=raw_table,
            key_column=key_column,
            symbol_ids=symbol_ids,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        if not grouped_ranges:
            return summary

        summary.checked_symbols = len(grouped_ranges)
        summary.checked_rows = int(sum(int(row["row_count"]) for row in grouped_ranges))

        actual_dates_by_key = _load_actual_dates_by_key(
            conn,
            raw_table=raw_table,
            key_column=key_column,
            symbol_ids=symbol_ids,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        summary.missing_sessions_total, summary.missing_sessions_by_symbol = _compute_missing_sessions(
            grouped_ranges,
            actual_dates_by_key,
            calendar_name=calendar_name,
        )

        invalid_keys = _load_invalid_row_keys(
            conn,
            raw_table=raw_table,
            key_column=key_column,
            symbol_ids=symbol_ids,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        summary.invalid_rows = len(invalid_keys)

    with db_engine.begin() as conn:
        raw_table = _reflect_raw_table(conn)
        key_column = _resolve_key_column(raw_table)
        _set_qc_status_pass(
            conn,
            raw_table=raw_table,
            key_column=key_column,
            symbol_ids=symbol_ids,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )
        _mark_invalid_rows_fail(
            conn,
            raw_table=raw_table,
            key_column=key_column,
            invalid_keys=invalid_keys,
        )

    return summary


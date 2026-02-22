from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Sequence

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..database import create_db_engine
from ..models.meta import Symbol


DEFAULT_HISTORY_START_DATE = date(2000, 1, 1)
RawMode = Literal["modern_symbol_id", "legacy_ticker"]


@dataclass(slots=True)
class IngestSchemaInfo:
    raw_mode: RawMode
    raw_columns: set[str]
    job_run_columns: set[str]


@dataclass(slots=True)
class IngestSummary:
    job_run_id: str
    full_refresh: bool
    requested_tickers: list[str]
    processed_tickers: int = 0
    total_rows_upserted: int = 0
    empty_tickers: list[str] = field(default_factory=list)
    failed_tickers: list[dict[str, str]] = field(default_factory=list)
    touched_symbol_ids: list[int] = field(default_factory=list)
    touched_tickers: list[str] = field(default_factory=list)
    min_trade_date: date | None = None
    max_trade_date: date | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_run_id": self.job_run_id,
            "full_refresh": self.full_refresh,
            "requested_tickers": self.requested_tickers,
            "processed_tickers": self.processed_tickers,
            "total_rows_upserted": self.total_rows_upserted,
            "empty_tickers": self.empty_tickers,
            "failed_tickers": self.failed_tickers,
            "touched_symbol_ids": self.touched_symbol_ids,
            "touched_tickers": self.touched_tickers,
            "min_trade_date": self.min_trade_date.isoformat() if self.min_trade_date else None,
            "max_trade_date": self.max_trade_date.isoformat() if self.max_trade_date else None,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coalesce_date(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _chunked(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for idx in range(0, len(rows), size):
        yield list(rows[idx : idx + size])


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


def _reflect_table(conn: sa.Connection, *, schema: str, table_name: str) -> sa.Table:
    return sa.Table(table_name, sa.MetaData(), schema=schema, autoload_with=conn)


def _validate_ingest_schema(engine: Engine) -> IngestSchemaInfo:
    inspector = sa.inspect(engine)
    try:
        raw_columns = {col["name"] for col in inspector.get_columns("ohlcv_daily", schema="raw")}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing required table raw.ohlcv_daily. Run `python -m src.cli init-db` first."
        ) from exc

    if not inspector.has_table("job_run", schema="meta"):
        raise RuntimeError(
            "meta.job_run table is missing. Run `python -m src.cli init-db` (or apply migration) before ingest."
        )
    job_run_columns = {col["name"] for col in inspector.get_columns("job_run", schema="meta")}

    if "job_run_id" not in raw_columns:
        raise RuntimeError("raw.ohlcv_daily.job_run_id column is required for lineage tracking.")

    if {"symbol_id", "trade_date"}.issubset(raw_columns):
        raw_mode: RawMode = "modern_symbol_id"
    elif {"ticker", "trade_date"}.issubset(raw_columns):
        raw_mode = "legacy_ticker"
    else:
        raise RuntimeError(
            "raw.ohlcv_daily must include either (symbol_id, trade_date) or (ticker, trade_date)."
        )

    return IngestSchemaInfo(raw_mode=raw_mode, raw_columns=raw_columns, job_run_columns=job_run_columns)


def _compact_message(obj: dict[str, Any], *, prefix: str, limit: int = 1000) -> str:
    message = f"{prefix} {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}"
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _create_job_run(
    engine: Engine,
    *,
    full_refresh: bool,
    requested_tickers: Sequence[str],
    start_date: date | None,
    end_date: date | None,
) -> str:
    now = _utcnow()
    run_params = {
        "full_refresh": bool(full_refresh),
        "requested_tickers": list(requested_tickers),
        "start_date": _coalesce_date(start_date),
        "end_date": _coalesce_date(end_date),
    }

    with engine.begin() as conn:
        job_table = _reflect_table(conn, schema="meta", table_name="job_run")
        cols = set(job_table.c.keys())

        payload: dict[str, Any] = {}
        if "job_name" in cols:
            payload["job_name"] = "raw_daily_ingest"
        if "status" in cols:
            payload["status"] = "RUNNING"
        if "started_at" in cols:
            payload["started_at"] = now
        if "run_params" in cols:
            payload["run_params"] = run_params
        elif "message" in cols:
            # Legacy schema fallback when JSONB columns are unavailable.
            payload["message"] = _compact_message({"run_params": run_params}, prefix="INGEST_START")

        insert_stmt = sa.insert(job_table).values(payload)
        if "id" not in cols:
            raise RuntimeError("meta.job_run.id column is required.")
        inserted = conn.execute(insert_stmt.returning(job_table.c.id)).scalar_one()
    return str(inserted)


def _finish_job_run(
    engine: Engine,
    *,
    job_run_id: str,
    status: str,
    run_stats: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with engine.begin() as conn:
        job_table = _reflect_table(conn, schema="meta", table_name="job_run")
        cols = set(job_table.c.keys())
        values: dict[str, Any] = {}

        if "status" in cols:
            values["status"] = status.upper()
        if "finished_at" in cols:
            values["finished_at"] = _utcnow()
        if "run_stats" in cols:
            values["run_stats"] = run_stats
        if "error_message" in cols:
            values["error_message"] = error_message
        elif "message" in cols:
            compact = {
                "status": status,
                "rows": run_stats.get("total_rows_upserted"),
                "processed_tickers": run_stats.get("processed_tickers"),
                "failed_tickers": len(run_stats.get("failed_tickers", []) or []),
            }
            if error_message:
                compact["error"] = error_message
            values["message"] = _compact_message(compact, prefix="INGEST_DONE")

        if not values:
            return

        conn.execute(
            sa.update(job_table)
            .where(job_table.c.id == job_run_id)
            .values(values)
        )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _fetch_yfinance_history(
    ticker: str,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    import yfinance as yf

    yf_end_exclusive = end_date + timedelta(days=1)
    history = yf.Ticker(ticker).history(
        start=start_date.isoformat(),
        end=yf_end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
    )
    if history is None:
        return pd.DataFrame()
    return history


def _load_target_symbols(conn: sa.Connection, tickers: Sequence[str] | None) -> list[dict[str, Any]]:
    symbol_table = Symbol.__table__
    stmt = (
        sa.select(symbol_table.c.id, symbol_table.c.ticker)
        .where(symbol_table.c.is_active.is_(True))
        .order_by(symbol_table.c.ticker.asc())
    )
    normalized = _normalize_tickers(tickers)
    if normalized:
        stmt = stmt.where(symbol_table.c.ticker.in_(normalized))
    rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]


def _load_last_trade_dates_by_symbol(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    symbol_ids: Sequence[int],
) -> dict[int, date]:
    if not symbol_ids:
        return {}
    stmt = (
        sa.select(raw_table.c.symbol_id, sa.func.max(raw_table.c.trade_date).label("max_trade_date"))
        .where(raw_table.c.symbol_id.in_(list(symbol_ids)))
        .group_by(raw_table.c.symbol_id)
    )
    return {
        int(row.symbol_id): row.max_trade_date
        for row in conn.execute(stmt)
        if row.max_trade_date is not None
    }


def _load_last_trade_dates_by_ticker(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    tickers: Sequence[str],
) -> dict[str, date]:
    normalized = _normalize_tickers(tickers)
    if not normalized:
        return {}
    stmt = (
        sa.select(raw_table.c.ticker, sa.func.max(raw_table.c.trade_date).label("max_trade_date"))
        .where(raw_table.c.ticker.in_(normalized))
        .group_by(raw_table.c.ticker)
    )
    return {
        str(row.ticker).upper(): row.max_trade_date
        for row in conn.execute(stmt)
        if row.max_trade_date is not None
    }


def _compute_fetch_range(
    *,
    last_trade_date: date | None,
    full_refresh: bool,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> tuple[date, date] | None:
    effective_end = requested_end_date or date.today()
    if full_refresh:
        effective_start = requested_start_date or DEFAULT_HISTORY_START_DATE
    else:
        incremental_start = (
            last_trade_date + timedelta(days=1) if last_trade_date is not None else DEFAULT_HISTORY_START_DATE
        )
        effective_start = incremental_start if requested_start_date is None else max(requested_start_date, incremental_start)
    if effective_start > effective_end:
        return None
    return effective_start, effective_end


def _normalize_history_base_rows(history: pd.DataFrame) -> list[dict[str, Any]]:
    if history.empty:
        return []

    frame = history.copy().reset_index()

    datetime_col = None
    for candidate in ("Date", "Datetime"):
        if candidate in frame.columns:
            datetime_col = candidate
            break
    if datetime_col is None:
        datetime_col = str(frame.columns[0])

    dt_series = pd.to_datetime(frame[datetime_col], errors="coerce")
    if getattr(dt_series.dt, "tz", None) is not None:
        dt_series = dt_series.dt.tz_convert(None)
    frame["trade_date"] = dt_series.dt.date

    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )

    for col in ("adj_close", "dividends", "stock_splits"):
        if col not in frame.columns:
            frame[col] = None

    frame = frame[
        [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
        ]
    ]
    # Convert NaN/NaT to None before DB insert.
    frame = frame.astype(object).where(pd.notna(frame), None)

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        trade_date = row.get("trade_date")
        if trade_date is None:
            continue

        volume_value = row.get("volume")
        if volume_value is not None:
            try:
                volume_value = int(volume_value)
            except (TypeError, ValueError):
                volume_value = None

        rows.append(
            {
                "trade_date": trade_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "adj_close": row.get("adj_close"),
                "volume": volume_value,
                "dividends": row.get("dividends") if row.get("dividends") is not None else 0.0,
                "stock_splits": row.get("stock_splits") if row.get("stock_splits") is not None else 0.0,
            }
        )
    return rows


def _build_modern_raw_rows(
    base_rows: Sequence[dict[str, Any]],
    *,
    symbol_id: int,
    job_run_id: str,
    raw_columns: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in base_rows:
        record = {
            "symbol_id": symbol_id,
            "trade_date": row["trade_date"],
            "job_run_id": job_run_id,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "adj_close": row.get("adj_close"),
            "volume": row.get("volume"),
            "dividends": row.get("dividends"),
            "stock_splits": row.get("stock_splits"),
            "source": "yfinance",
            "qc_status": "pending",
            "qc_meta": {},
        }
        records.append({k: v for k, v in record.items() if k in raw_columns})
    return records


def _build_legacy_raw_rows(
    base_rows: Sequence[dict[str, Any]],
    *,
    ticker: str,
    job_run_id: str,
    raw_columns: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in base_rows:
        record = {
            "ticker": ticker,
            "trade_date": row["trade_date"],
            "job_run_id": job_run_id,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
        }
        records.append({k: v for k, v in record.items() if k in raw_columns})
    return records


def _delete_existing_rows_for_range(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    raw_mode: RawMode,
    key_value: int | str,
    start_date: date,
    end_date: date,
) -> int:
    stmt = sa.delete(raw_table)
    if raw_mode == "modern_symbol_id":
        stmt = stmt.where(raw_table.c.symbol_id == key_value)
    else:
        stmt = stmt.where(raw_table.c.ticker == key_value)
    stmt = stmt.where(raw_table.c.trade_date >= start_date).where(raw_table.c.trade_date <= end_date)
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def _batch_upsert_modern_daily_ohlcv(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    rows: Sequence[dict[str, Any]],
    chunk_size: int = 1000,
) -> int:
    if not rows:
        return 0
    total = 0
    for chunk in _chunked(rows, chunk_size):
        stmt = pg_insert(raw_table).values(chunk)
        update_columns = {
            column.name: getattr(stmt.excluded, column.name)
            for column in raw_table.columns
            if column.name not in {"symbol_id", "trade_date"}
        }
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=[raw_table.c.symbol_id, raw_table.c.trade_date],
            set_=update_columns,
        )
        result = conn.execute(upsert_stmt)
        total += int(result.rowcount or 0)
    return total


def _batch_insert_legacy_daily_ohlcv(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    rows: Sequence[dict[str, Any]],
    chunk_size: int = 1000,
) -> int:
    if not rows:
        return 0
    total = 0
    for chunk in _chunked(rows, chunk_size):
        result = conn.execute(sa.insert(raw_table).values(chunk))
        total += int(result.rowcount or 0)
    return total


def _update_summary_dates(summary: IngestSummary, records: Sequence[dict[str, Any]]) -> None:
    record_dates = [r["trade_date"] for r in records if r.get("trade_date") is not None]
    if not record_dates:
        return
    min_d = min(record_dates)
    max_d = max(record_dates)
    summary.min_trade_date = min_d if summary.min_trade_date is None else min(summary.min_trade_date, min_d)
    summary.max_trade_date = max_d if summary.max_trade_date is None else max(summary.max_trade_date, max_d)


def ingest_raw_daily_ohlcv(
    *,
    tickers: Sequence[str] | None = None,
    full_refresh: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
    echo: bool = False,
    engine: Engine | None = None,
) -> IngestSummary:
    db_engine = engine or create_db_engine(echo=echo)
    schema_info = _validate_ingest_schema(db_engine)
    normalized_tickers = _normalize_tickers(tickers)

    job_run_id = _create_job_run(
        db_engine,
        full_refresh=full_refresh,
        requested_tickers=normalized_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    summary = IngestSummary(
        job_run_id=job_run_id,
        full_refresh=bool(full_refresh),
        requested_tickers=normalized_tickers,
    )

    try:
        with db_engine.connect() as conn:
            raw_table = _reflect_table(conn, schema="raw", table_name="ohlcv_daily")

            if schema_info.raw_mode == "modern_symbol_id":
                symbols = _load_target_symbols(conn, normalized_tickers)
                if normalized_tickers and len(symbols) != len(normalized_tickers):
                    found = {str(row["ticker"]).upper() for row in symbols}
                    for ticker in normalized_tickers:
                        if ticker not in found:
                            summary.failed_tickers.append(
                                {"ticker": ticker, "error": "Ticker not found in meta.symbols or inactive."}
                            )
                symbol_ids = [int(row["id"]) for row in symbols]
                last_trade_dates_by_symbol = _load_last_trade_dates_by_symbol(
                    conn,
                    raw_table=raw_table,
                    symbol_ids=symbol_ids,
                )
                targets = [
                    {
                        "ticker": str(row["ticker"]).upper(),
                        "symbol_id": int(row["id"]),
                        "last_trade_date": last_trade_dates_by_symbol.get(int(row["id"])),
                    }
                    for row in symbols
                ]
            else:
                if normalized_tickers:
                    target_tickers = normalized_tickers
                else:
                    symbols = _load_target_symbols(conn, None)
                    target_tickers = [str(row["ticker"]).upper() for row in symbols]
                last_trade_dates_by_ticker = _load_last_trade_dates_by_ticker(
                    conn,
                    raw_table=raw_table,
                    tickers=target_tickers,
                )
                targets = [
                    {
                        "ticker": ticker,
                        "symbol_id": None,
                        "last_trade_date": last_trade_dates_by_ticker.get(ticker),
                    }
                    for ticker in target_tickers
                ]

        for target in targets:
            ticker = str(target["ticker"]).upper()
            symbol_id = target["symbol_id"]
            last_trade_date = target["last_trade_date"]

            date_range = _compute_fetch_range(
                last_trade_date=last_trade_date,
                full_refresh=full_refresh,
                requested_start_date=start_date,
                requested_end_date=end_date,
            )
            if date_range is None:
                summary.empty_tickers.append(ticker)
                continue
            fetch_start, fetch_end = date_range

            try:
                history = _fetch_yfinance_history(ticker, start_date=fetch_start, end_date=fetch_end)
                base_rows = _normalize_history_base_rows(history)
            except Exception as exc:  # noqa: BLE001
                summary.failed_tickers.append({"ticker": ticker, "error": str(exc)})
                continue

            summary.processed_tickers += 1

            if schema_info.raw_mode == "modern_symbol_id":
                if symbol_id is None:
                    summary.failed_tickers.append(
                        {"ticker": ticker, "error": "symbol_id is required for modern raw schema."}
                    )
                    continue
                records = _build_modern_raw_rows(
                    base_rows,
                    symbol_id=int(symbol_id),
                    job_run_id=job_run_id,
                    raw_columns=schema_info.raw_columns,
                )
            else:
                records = _build_legacy_raw_rows(
                    base_rows,
                    ticker=ticker,
                    job_run_id=job_run_id,
                    raw_columns=schema_info.raw_columns,
                )

            if not records:
                summary.empty_tickers.append(ticker)
                continue

            with db_engine.begin() as conn:
                raw_table = _reflect_table(conn, schema="raw", table_name="ohlcv_daily")
                if schema_info.raw_mode == "legacy_ticker":
                    # Legacy table has no natural-key unique constraint, so replace rows in-range.
                    _delete_existing_rows_for_range(
                        conn,
                        raw_table=raw_table,
                        raw_mode=schema_info.raw_mode,
                        key_value=ticker,
                        start_date=fetch_start,
                        end_date=fetch_end,
                    )
                    affected = _batch_insert_legacy_daily_ohlcv(conn, raw_table=raw_table, rows=records)
                else:
                    if full_refresh:
                        _delete_existing_rows_for_range(
                            conn,
                            raw_table=raw_table,
                            raw_mode=schema_info.raw_mode,
                            key_value=int(symbol_id),
                            start_date=fetch_start,
                            end_date=fetch_end,
                        )
                    affected = _batch_upsert_modern_daily_ohlcv(conn, raw_table=raw_table, rows=records)

            summary.total_rows_upserted += affected
            summary.touched_tickers.append(ticker)
            if symbol_id is not None:
                summary.touched_symbol_ids.append(int(symbol_id))
            _update_summary_dates(summary, records)

        summary.touched_symbol_ids = sorted(set(summary.touched_symbol_ids))
        summary.touched_tickers = sorted(set(summary.touched_tickers))
        status = "success" if not summary.failed_tickers else "partial_success"
        _finish_job_run(db_engine, job_run_id=job_run_id, status=status, run_stats=summary.as_dict())
        return summary
    except Exception as exc:  # noqa: BLE001
        _finish_job_run(
            db_engine,
            job_run_id=job_run_id,
            status="failed",
            run_stats=summary.as_dict(),
            error_message=str(exc),
        )
        raise


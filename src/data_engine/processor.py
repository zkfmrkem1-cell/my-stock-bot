from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from ..database import create_db_engine


RawMode = Literal["modern_symbol_id", "legacy_ticker"]


@dataclass(slots=True)
class ProcessSummary:
    requested_tickers: list[str]
    processed_tickers: list[str] = field(default_factory=list)
    processed_symbols: int = 0
    raw_rows_read: int = 0
    feature_rows_upserted: int = 0
    label_rows_upserted: int = 0
    min_trade_date: date | None = None
    max_trade_date: date | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_tickers": self.requested_tickers,
            "processed_tickers": self.processed_tickers,
            "processed_symbols": self.processed_symbols,
            "raw_rows_read": self.raw_rows_read,
            "feature_rows_upserted": self.feature_rows_upserted,
            "label_rows_upserted": self.label_rows_upserted,
            "min_trade_date": self.min_trade_date.isoformat() if self.min_trade_date else None,
            "max_trade_date": self.max_trade_date.isoformat() if self.max_trade_date else None,
            "warnings": self.warnings,
        }


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


def _ticker_matches_market(ticker: str, market: str) -> bool:
    scope = str(market or "all").lower()
    t = str(ticker or "").upper()
    if scope == "kr":
        return t.endswith(".KS")
    if scope == "us":
        return not t.endswith(".KS")
    return True


def _filter_tickers_by_market(tickers: Sequence[str], market: str) -> list[str]:
    return [t for t in _normalize_tickers(tickers) if _ticker_matches_market(t, market)]


def _reflect_table(conn: sa.Connection, *, schema: str, table_name: str) -> sa.Table:
    return sa.Table(table_name, sa.MetaData(), schema=schema, autoload_with=conn)


def _require_table(conn: sa.Connection, *, schema: str, table_name: str) -> sa.Table:
    inspector = sa.inspect(conn)
    if not inspector.has_table(table_name, schema=schema):
        raise RuntimeError(f"Missing required table {schema}.{table_name}. Run `python -m src.cli init-db` first.")
    return _reflect_table(conn, schema=schema, table_name=table_name)


def _load_meta_tickers_for_market(conn: sa.Connection, market: str) -> list[str]:
    scope = str(market or "all").lower()
    meta_symbols = _require_table(conn, schema="meta", table_name="symbols")
    stmt = (
        sa.select(meta_symbols.c.ticker)
        .where(meta_symbols.c.is_active.is_(True))
        .order_by(meta_symbols.c.ticker.asc())
    )
    if scope == "kr":
        stmt = stmt.where(meta_symbols.c.ticker.like("%.KS"))
    elif scope == "us":
        stmt = stmt.where(sa.not_(meta_symbols.c.ticker.like("%.KS")))
    return [str(r.ticker).upper() for r in conn.execute(stmt)]


def _detect_raw_mode(raw_table: sa.Table) -> RawMode:
    cols = set(raw_table.c.keys())
    if {"symbol_id", "trade_date"}.issubset(cols):
        return "modern_symbol_id"
    if {"ticker", "trade_date"}.issubset(cols):
        return "legacy_ticker"
    raise RuntimeError("raw.ohlcv_daily must include either (symbol_id, trade_date) or (ticker, trade_date).")


def _chunked(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield list(rows[i : i + size])


def _ensure_symbol_mapping(
    conn: sa.Connection,
    *,
    requested_tickers: Sequence[str],
    raw_tickers: Sequence[str] | None = None,
) -> dict[str, int]:
    meta_symbols = _require_table(conn, schema="meta", table_name="symbols")

    target_tickers = _normalize_tickers(requested_tickers if requested_tickers else raw_tickers)
    if not target_tickers:
        return {}

    # Auto-seed minimal symbol rows when raw data exists but meta.symbols is empty.
    insert_rows = [{"ticker": t} for t in target_tickers]
    stmt = pg_insert(meta_symbols).values(insert_rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=[meta_symbols.c.ticker])
    conn.execute(stmt)

    rows = conn.execute(
        sa.select(meta_symbols.c.id, meta_symbols.c.ticker)
        .where(meta_symbols.c.ticker.in_(target_tickers))
        .order_by(meta_symbols.c.ticker.asc())
    ).all()
    return {str(r.ticker).upper(): int(r.id) for r in rows}


def _load_raw_tickers_from_table(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    start_date: date | None,
    end_date: date | None,
) -> list[str]:
    if "ticker" not in raw_table.c:
        return []
    stmt = sa.select(raw_table.c.ticker).distinct().order_by(raw_table.c.ticker.asc())
    if start_date is not None:
        stmt = stmt.where(raw_table.c.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(raw_table.c.trade_date <= end_date)
    return [str(r.ticker).upper() for r in conn.execute(stmt)]


def _load_raw_dataframe(
    conn: sa.Connection,
    *,
    raw_table: sa.Table,
    raw_mode: RawMode,
    ticker_to_symbol_id: dict[str, int],
    requested_tickers: Sequence[str],
    start_date: date | None,
    end_date: date | None,
) -> tuple[pd.DataFrame, list[str]]:
    cols = set(raw_table.c.keys())
    requested = _normalize_tickers(requested_tickers)

    select_cols: list[sa.ColumnElement[Any]] = []
    if raw_mode == "modern_symbol_id":
        select_cols.extend([raw_table.c.symbol_id, raw_table.c.trade_date])
    else:
        select_cols.extend([raw_table.c.ticker, raw_table.c.trade_date])

    for name in ("adj_close", "close", "volume", "created_at"):
        if name in cols:
            select_cols.append(raw_table.c[name])

    stmt = sa.select(*select_cols)
    if start_date is not None:
        stmt = stmt.where(raw_table.c.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(raw_table.c.trade_date <= end_date)

    if raw_mode == "modern_symbol_id":
        if requested:
            symbol_ids = [ticker_to_symbol_id[t] for t in requested if t in ticker_to_symbol_id]
            if not symbol_ids:
                return pd.DataFrame(), []
            stmt = stmt.where(raw_table.c.symbol_id.in_(symbol_ids))
        if "created_at" in cols:
            stmt = stmt.order_by(raw_table.c.symbol_id.asc(), raw_table.c.trade_date.asc(), raw_table.c.created_at.asc())
        else:
            stmt = stmt.order_by(raw_table.c.symbol_id.asc(), raw_table.c.trade_date.asc())
    else:
        if requested:
            stmt = stmt.where(raw_table.c.ticker.in_(requested))
        if "created_at" in cols:
            stmt = stmt.order_by(raw_table.c.ticker.asc(), raw_table.c.trade_date.asc(), raw_table.c.created_at.asc())
        else:
            stmt = stmt.order_by(raw_table.c.ticker.asc(), raw_table.c.trade_date.asc())

    rows = conn.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)
    if raw_mode == "legacy_ticker":
        df["ticker"] = df["ticker"].astype(str).str.upper()
        if "created_at" in df.columns:
            df = df.sort_values(["ticker", "trade_date", "created_at"])
        else:
            df = df.sort_values(["ticker", "trade_date"])
        df = df.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
        df["symbol_id"] = df["ticker"].map(ticker_to_symbol_id)
        df = df[df["symbol_id"].notna()].copy()
        df["symbol_id"] = df["symbol_id"].astype("int64")
        used_tickers = sorted(df["ticker"].astype(str).str.upper().unique().tolist())
    else:
        inverse_map = {v: k for k, v in ticker_to_symbol_id.items()}
        df["ticker"] = df["symbol_id"].map(inverse_map)
        if "created_at" in df.columns:
            df = df.sort_values(["symbol_id", "trade_date", "created_at"])
        else:
            df = df.sort_values(["symbol_id", "trade_date"])
        df = df.drop_duplicates(subset=["symbol_id", "trade_date"], keep="last")
        used_tickers = sorted([t for t in df["ticker"].dropna().astype(str).str.upper().unique().tolist()])

    if "adj_close" not in df.columns:
        df["adj_close"] = np.nan
    if "close" not in df.columns:
        raise RuntimeError("raw.ohlcv_daily.close column is required for processing.")
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["price_base"] = pd.to_numeric(df["adj_close"], errors="coerce").combine_first(
        pd.to_numeric(df["close"], errors="coerce")
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["symbol_id", "trade_date"]).reset_index(drop=True)

    return df, used_tickers


def _calc_rsi14(price: pd.Series) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # If avg_loss is zero and avg_gain exists, RSI should be 100.
    rsi = rsi.where(~((avg_loss == 0) & avg_gain.notna()), 100.0)
    return rsi


def _compute_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol_id, g in raw_df.groupby("symbol_id", sort=True):
        group = g.sort_values("trade_date").copy()
        price = group["price_base"]
        volume = group["volume"]

        ma_5 = price.rolling(window=5, min_periods=5).mean()
        ma_20 = price.rolling(window=20, min_periods=20).mean()
        ma_25 = price.rolling(window=25, min_periods=25).mean()
        ma_60 = price.rolling(window=60, min_periods=60).mean()
        rsi_14 = _calc_rsi14(price)

        feat = pd.DataFrame(
            {
                "symbol_id": int(symbol_id),
                "trade_date": group["trade_date"].values,
                "close_price": price.values,
                "return_1d": price.pct_change(periods=1, fill_method=None).values,
                "return_5d": price.pct_change(periods=5, fill_method=None).values,
                "ma_5": ma_5.values,
                "ma_20": ma_20.values,
                "ma_60": ma_60.values,
                "dist_ma20": (price / ma_20 - 1.0).values,
                "rsi_14": rsi_14.values,
                "vol_ratio_20": (volume / volume.rolling(window=20, min_periods=20).mean()).values,
                "feature_version": "v1",
                "price_basis": np.where(group["adj_close"].notna(), "adj_close", "close"),
                "disparity25": (price / ma_25 * 100.0).values,
                "ma_25": ma_25.values,
            }
        )
        parts.append(feat)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _compute_future_max_drawdown(price: pd.Series, horizon: int = 20) -> pd.Series:
    arr = pd.to_numeric(price, errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        base = arr[i]
        if not np.isfinite(base) or base == 0:
            continue
        future = arr[i + 1 : i + 1 + horizon]
        if future.size == 0:
            continue
        rel = future / base - 1.0
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        out[i] = float(min(float(np.min(rel)), 0.0))
    return pd.Series(out, index=price.index)


def _compute_label_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol_id, g in raw_df.groupby("symbol_id", sort=True):
        group = g.sort_values("trade_date").copy()
        price = group["price_base"]

        fwd_1d = price.shift(-1) / price - 1.0
        fwd_5d = price.shift(-5) / price - 1.0
        fwd_20d = price.shift(-20) / price - 1.0
        max_dd_20d = _compute_future_max_drawdown(price, horizon=20)

        label = pd.DataFrame(
            {
                "symbol_id": int(symbol_id),
                "trade_date": group["trade_date"].values,
                "fwd_return_1d": fwd_1d.values,
                "fwd_return_5d": fwd_5d.values,
                "fwd_return_20d": fwd_20d.values,
                "target_up_5d": np.where(pd.isna(fwd_5d), None, fwd_5d > 0),
                "target_up_20d": np.where(pd.isna(fwd_20d), None, fwd_20d > 0),
                "target_rebound_after_oversold": None,
                "label_version": "v1",
                "price_basis": np.where(group["adj_close"].notna(), "adj_close", "close"),
                "fwd_ret_5d": fwd_5d.values,
                "max_dd_20d": max_dd_20d.values,
                "fwd_ret_20d": fwd_20d.values,
            }
        )
        parts.append(label)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _clean_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _nan_to_none_records(df: pd.DataFrame, *, table_columns: set[str], meta_builder: str) -> list[dict[str, Any]]:
    if df.empty:
        return []

    working = df.copy()
    if meta_builder == "feature":
        working["feature_meta"] = [
            {
                "disparity25": _clean_json_scalar(row["disparity25"]),
                "ma_25": _clean_json_scalar(row["ma_25"]),
                "price_basis": _clean_json_scalar(row["price_basis"]),
            }
            for _, row in working[["disparity25", "ma_25", "price_basis"]].iterrows()
        ]
    elif meta_builder == "label":
        working["label_meta"] = [
            {
                "fwd_ret_5d": _clean_json_scalar(row["fwd_ret_5d"]),
                "fwd_ret_20d": _clean_json_scalar(row["fwd_ret_20d"]),
                "max_dd_20d": _clean_json_scalar(row["max_dd_20d"]),
                "price_basis": _clean_json_scalar(row["price_basis"]),
            }
            for _, row in working[["fwd_ret_5d", "fwd_ret_20d", "max_dd_20d", "price_basis"]].iterrows()
        ]

    drop_cols = {"price_basis", "disparity25", "ma_25", "fwd_ret_5d", "fwd_ret_20d", "max_dd_20d"}
    working = working[[c for c in working.columns if c not in drop_cols and c in table_columns]]

    # Convert NaN/NaT to None before DB insert.
    working = working.astype(object).where(pd.notna(working), None)
    return working.to_dict(orient="records")


def _batch_upsert(
    conn: sa.Connection,
    *,
    table: sa.Table,
    rows: Sequence[dict[str, Any]],
    pk_columns: Sequence[str],
    chunk_size: int = 1000,
) -> int:
    if not rows:
        return 0
    total = 0
    pk_set = set(pk_columns)
    for chunk in _chunked(rows, chunk_size):
        stmt = pg_insert(table).values(chunk)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in table.columns
            if c.name not in pk_set and c.name != "created_at"
        }
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=[table.c[name] for name in pk_columns],
            set_=update_cols,
        )
        result = conn.execute(upsert_stmt)
        total += int(result.rowcount or 0)
    return total


def process_feature_and_label_data(
    *,
    tickers: Sequence[str] | None = None,
    market: str = "all",
    start_date: date | None = None,
    end_date: date | None = None,
    skip_labels: bool = False,
    echo: bool = False,
    engine: Engine | None = None,
) -> ProcessSummary:
    db_engine = engine or create_db_engine(echo=echo)
    market_scope = str(market or "all").lower()
    normalized_tickers = _normalize_tickers(tickers)
    filtered_requested_tickers = (
        _filter_tickers_by_market(normalized_tickers, market_scope)
        if market_scope != "all"
        else normalized_tickers
    )
    summary = ProcessSummary(requested_tickers=normalized_tickers)
    if market_scope != "all" and normalized_tickers and len(filtered_requested_tickers) != len(normalized_tickers):
        dropped = sorted(set(normalized_tickers) - set(filtered_requested_tickers))
        if dropped:
            preview = ",".join(dropped[:10])
            suffix = "" if len(dropped) <= 10 else f",... ({len(dropped)} total)"
            summary.warnings.append(f"Ignored {len(dropped)} ticker(s) outside market='{market_scope}': {preview}{suffix}")
    summary.requested_tickers = filtered_requested_tickers

    with db_engine.begin() as conn:
        raw_table = _require_table(conn, schema="raw", table_name="ohlcv_daily")
        _require_table(conn, schema="feat", table_name="daily_features")
        _require_table(conn, schema="label", table_name="daily_labels")

        raw_mode = _detect_raw_mode(raw_table)
        effective_requested_tickers = filtered_requested_tickers
        if not effective_requested_tickers and market_scope != "all":
            effective_requested_tickers = _load_meta_tickers_for_market(conn, market_scope)
        raw_tickers = []
        if raw_mode == "legacy_ticker":
            raw_tickers = _load_raw_tickers_from_table(
                conn,
                raw_table=raw_table,
                start_date=start_date,
                end_date=end_date,
            )
            if market_scope != "all":
                raw_tickers = _filter_tickers_by_market(raw_tickers, market_scope)
        ticker_to_symbol_id = _ensure_symbol_mapping(
            conn,
            requested_tickers=effective_requested_tickers,
            raw_tickers=raw_tickers,
        )

        raw_df, used_tickers = _load_raw_dataframe(
            conn,
            raw_table=raw_table,
            raw_mode=raw_mode,
            ticker_to_symbol_id=ticker_to_symbol_id,
            requested_tickers=effective_requested_tickers,
            start_date=start_date,
            end_date=end_date,
        )

    if raw_df.empty:
        summary.warnings.append("No raw rows found for the requested scope.")
        return summary

    summary.raw_rows_read = int(len(raw_df))
    summary.processed_tickers = used_tickers
    summary.processed_symbols = int(raw_df["symbol_id"].nunique())
    summary.min_trade_date = min(raw_df["trade_date"])
    summary.max_trade_date = max(raw_df["trade_date"])
    if raw_df["adj_close"].isna().all():
        summary.warnings.append("raw.adj_close is unavailable in current raw schema; fallback to close was used.")

    feature_df = _compute_feature_frame(raw_df)

    with db_engine.begin() as conn:
        feat_table = _require_table(conn, schema="feat", table_name="daily_features")

        feature_rows = _nan_to_none_records(
            feature_df,
            table_columns=set(feat_table.c.keys()),
            meta_builder="feature",
        )

        summary.feature_rows_upserted = _batch_upsert(
            conn,
            table=feat_table,
            rows=feature_rows,
            pk_columns=("symbol_id", "trade_date"),
        )

        if not skip_labels:
            label_df = _compute_label_frame(raw_df)
            label_table = _require_table(conn, schema="label", table_name="daily_labels")
            label_rows = _nan_to_none_records(
                label_df,
                table_columns=set(label_table.c.keys()),
                meta_builder="label",
            )
            summary.label_rows_upserted = _batch_upsert(
                conn,
                table=label_table,
                rows=label_rows,
                pk_columns=("symbol_id", "trade_date"),
            )

    return summary

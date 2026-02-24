from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Sequence
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from ..database import _load_dotenv_if_available, create_db_engine


@dataclass(slots=True)
class SymbolReportBuildSummary:
    report_date: date
    processed_symbols: int = 0
    rows_written: int = 0
    warnings: list[str] = field(default_factory=list)


def _f(value: Any, *, ndigits: int | None = None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if ndigits is None:
        return v
    return round(v, ndigits)


def _pct(value: Any, *, ndigits: int = 2) -> float | None:
    v = _f(value)
    if v is None:
        return None
    return round(v * 100.0, ndigits)


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _rule_flags(*, rsi_14: float | None, dist_ma20_pct: float | None, vol_ratio_20: float | None) -> list[str]:
    flags: list[str] = []
    if rsi_14 is not None:
        if rsi_14 <= 30:
            flags.append("rsi_oversold")
        elif rsi_14 >= 70:
            flags.append("rsi_overbought")
    if dist_ma20_pct is not None:
        if dist_ma20_pct <= -5:
            flags.append("below_ma20_deep")
        elif dist_ma20_pct >= 5:
            flags.append("above_ma20_extended")
    if vol_ratio_20 is not None and vol_ratio_20 >= 2:
        flags.append("volume_spike")
    return flags


def _build_summary_text(
    *,
    ticker: str,
    report_date: date,
    close_price: float | None,
    return_1d_pct: float | None,
    return_5d_pct: float | None,
    rsi_14: float | None,
    dist_ma20_pct: float | None,
    vol_ratio_20: float | None,
    news_count: int,
    caution_news_count: int,
    flags: Sequence[str],
) -> str:
    parts = [f"{ticker} {report_date.isoformat()} snapshot"]
    if close_price is not None:
        parts.append(f"close={close_price:.4f}")
    if return_1d_pct is not None:
        parts.append(f"1d={return_1d_pct:+.2f}%")
    if return_5d_pct is not None:
        parts.append(f"5d={return_5d_pct:+.2f}%")
    if rsi_14 is not None:
        parts.append(f"RSI14={rsi_14:.1f}")
    if dist_ma20_pct is not None:
        parts.append(f"distMA20={dist_ma20_pct:+.2f}%")
    if vol_ratio_20 is not None:
        parts.append(f"volRatio20={vol_ratio_20:.2f}x")
    parts.append(f"news7d={news_count}")
    if caution_news_count:
        parts.append(f"cautionNews={caution_news_count}")
    if flags:
        parts.append("flags=" + ",".join(flags))
    return " | ".join(parts)


def _load_target_report_date(conn: sa.Connection, explicit_date: date | None) -> date:
    if explicit_date is not None:
        return explicit_date
    feat_t = sa.Table("daily_features", sa.MetaData(), schema="feat", autoload_with=conn)
    latest = conn.execute(sa.select(sa.func.max(feat_t.c.trade_date))).scalar_one_or_none()
    if latest is None:
        raise RuntimeError("feat.daily_features is empty. Run process first.")
    return latest


def _load_feature_rows(
    conn: sa.Connection,
    *,
    report_date: date,
    tickers: Sequence[str] | None = None,
    market: str = "all",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    meta_t = sa.Table("symbols", sa.MetaData(), schema="meta", autoload_with=conn)
    feat_t = sa.Table("daily_features", sa.MetaData(), schema="feat", autoload_with=conn)

    stmt = (
        sa.select(
            meta_t.c.id.label("symbol_id"),
            meta_t.c.ticker,
            meta_t.c.name,
            meta_t.c.exchange,
            meta_t.c.market,
            meta_t.c.currency,
            meta_t.c.sector,
            meta_t.c.industry,
            feat_t.c.trade_date,
            feat_t.c.close_price,
            feat_t.c.return_1d,
            feat_t.c.return_5d,
            feat_t.c.ma_5,
            feat_t.c.ma_20,
            feat_t.c.ma_60,
            feat_t.c.dist_ma20,
            feat_t.c.rsi_14,
            feat_t.c.vol_ratio_20,
            feat_t.c.feature_version,
        )
        .join(feat_t, feat_t.c.symbol_id == meta_t.c.id)
        .where(meta_t.c.is_active == True, feat_t.c.trade_date == report_date)
        .order_by(meta_t.c.ticker.asc())
    )
    market_scope = str(market or "all").lower()
    if market_scope == "kr":
        stmt = stmt.where(meta_t.c.ticker.like("%.KS"))
    elif market_scope == "us":
        stmt = stmt.where(sa.not_(meta_t.c.ticker.like("%.KS")))
    if tickers:
        stmt = stmt.where(meta_t.c.ticker.in_(list(tickers)))
    if limit and limit > 0:
        stmt = stmt.limit(int(limit))
    return [dict(r) for r in conn.execute(stmt).mappings().all()]


def _load_news_by_symbol(
    conn: sa.Connection,
    *,
    symbol_ids: Sequence[int],
    start_ts: datetime,
    end_ts: datetime,
    max_news_refs_per_symbol: int,
) -> dict[int, list[dict[str, Any]]]:
    if not symbol_ids:
        return {}

    news_t = sa.Table("stock_news", sa.MetaData(), schema="news", autoload_with=conn)
    event_ts = sa.func.coalesce(news_t.c.published_at, news_t.c.created_at)
    stmt = (
        sa.select(
            news_t.c.id,
            news_t.c.symbol_id,
            news_t.c.title,
            news_t.c.source,
            news_t.c.sentiment_flag,
            news_t.c.published_at,
            news_t.c.created_at,
            event_ts.label("event_ts"),
        )
        .where(
            news_t.c.symbol_id.in_(list(symbol_ids)),
            event_ts >= start_ts,
            event_ts < end_ts,
        )
        .order_by(news_t.c.symbol_id.asc(), event_ts.desc(), news_t.c.created_at.desc())
    )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in conn.execute(stmt).mappings():
        symbol_id = int(row["symbol_id"])
        bucket = grouped.setdefault(symbol_id, [])
        if len(bucket) >= max_news_refs_per_symbol:
            continue
        published_at = row["published_at"]
        created_at = row["created_at"]
        bucket.append(
            {
                "id": str(row["id"]),
                "title": _safe_str(row["title"]) or "",
                "source": _safe_str(row["source"]) or "",
                "sentiment_flag": _safe_str(row["sentiment_flag"]) or "normal",
                "published_at": published_at.isoformat() if published_at else None,
                "created_at": created_at.isoformat() if created_at else None,
            }
        )
    return grouped


def build_symbol_reports(
    *,
    report_date: date | None = None,
    tickers: Sequence[str] | None = None,
    market: str = "all",
    news_lookback_days: int = 7,
    max_news_refs_per_symbol: int = 10,
    limit: int | None = None,
    echo: bool = False,
    engine: Engine | None = None,
) -> SymbolReportBuildSummary:
    _load_dotenv_if_available()
    db_engine = engine or create_db_engine(echo=echo)

    with db_engine.begin() as conn:
        insp = sa.inspect(conn)
        if not insp.has_table("symbol_reports", schema="report"):
            raise RuntimeError("report.symbol_reports table is missing. Run `python -m src.cli init-db` first.")

        resolved_report_date = _load_target_report_date(conn, report_date)
        feature_rows = _load_feature_rows(
            conn,
            report_date=resolved_report_date,
            tickers=tickers,
            market=market,
            limit=limit,
        )

        summary = SymbolReportBuildSummary(report_date=resolved_report_date)
        if not feature_rows:
            summary.warnings.append("No feature rows found for the selected report_date.")
            return summary

        symbol_ids = [int(r["symbol_id"]) for r in feature_rows]
        window_start = datetime.combine(
            resolved_report_date - timedelta(days=max(1, int(news_lookback_days)) - 1),
            time.min,
            tzinfo=timezone.utc,
        )
        window_end = datetime.combine(
            resolved_report_date + timedelta(days=1),
            time.min,
            tzinfo=timezone.utc,
        )
        news_by_symbol = _load_news_by_symbol(
            conn,
            symbol_ids=symbol_ids,
            start_ts=window_start,
            end_ts=window_end,
            max_news_refs_per_symbol=max_news_refs_per_symbol,
        )

        report_t = sa.Table("symbol_reports", sa.MetaData(), schema="report", autoload_with=conn)
        rows: list[dict[str, Any]] = []
        generated_at = datetime.now(timezone.utc).isoformat()

        for row in feature_rows:
            ticker = str(row["ticker"]).upper()
            symbol_id = int(row["symbol_id"])
            news_items = news_by_symbol.get(symbol_id, [])

            close_price = _f(row.get("close_price"), ndigits=6)
            return_1d = _f(row.get("return_1d"), ndigits=8)
            return_5d = _f(row.get("return_5d"), ndigits=8)
            rsi_14 = _f(row.get("rsi_14"), ndigits=4)
            dist_ma20 = _f(row.get("dist_ma20"), ndigits=8)
            vol_ratio_20 = _f(row.get("vol_ratio_20"), ndigits=6)

            return_1d_pct = _pct(row.get("return_1d"))
            return_5d_pct = _pct(row.get("return_5d"))
            dist_ma20_pct = _pct(row.get("dist_ma20"))

            caution_news_count = sum(1 for n in news_items if n.get("sentiment_flag") in {"caution", "exclude"})
            source_counts: dict[str, int] = {}
            for n in news_items:
                source = n.get("source") or "unknown"
                source_counts[source] = int(source_counts.get(source, 0)) + 1

            flags = _rule_flags(
                rsi_14=rsi_14,
                dist_ma20_pct=dist_ma20_pct,
                vol_ratio_20=vol_ratio_20,
            )

            feature_snapshot = {
                "feature_version": _safe_str(row.get("feature_version")) or "v1",
                "trade_date": str(row["trade_date"]),
                "close_price": close_price,
                "return_1d": return_1d,
                "return_5d": return_5d,
                "ma_5": _f(row.get("ma_5"), ndigits=6),
                "ma_20": _f(row.get("ma_20"), ndigits=6),
                "ma_60": _f(row.get("ma_60"), ndigits=6),
                "dist_ma20": dist_ma20,
                "rsi_14": rsi_14,
                "vol_ratio_20": vol_ratio_20,
            }

            metrics_payload = {
                "ticker": ticker,
                "symbol_name": _safe_str(row.get("name")),
                "exchange": _safe_str(row.get("exchange")),
                "market": _safe_str(row.get("market")),
                "currency": _safe_str(row.get("currency")),
                "sector": _safe_str(row.get("sector")),
                "industry": _safe_str(row.get("industry")),
                "return_1d_pct": return_1d_pct,
                "return_5d_pct": return_5d_pct,
                "dist_ma20_pct": dist_ma20_pct,
                "rsi_14": rsi_14,
                "vol_ratio_20": vol_ratio_20,
                "news_count_window": len(news_items),
                "caution_news_count": caution_news_count,
                "news_source_counts": source_counts,
                "flags": flags,
            }

            report_meta = {
                "builder": "symbol_report_builder.rule_v1",
                "generated_at_utc": generated_at,
                "news_window_start_utc": window_start.isoformat(),
                "news_window_end_utc": window_end.isoformat(),
                "news_headline_samples": [
                    {
                        "id": n.get("id"),
                        "source": n.get("source"),
                        "title": n.get("title"),
                        "published_at": n.get("published_at"),
                    }
                    for n in news_items[:3]
                ],
                "ai_enabled": False,
            }

            summary_text = _build_summary_text(
                ticker=ticker,
                report_date=resolved_report_date,
                close_price=close_price,
                return_1d_pct=return_1d_pct,
                return_5d_pct=return_5d_pct,
                rsi_14=rsi_14,
                dist_ma20_pct=dist_ma20_pct,
                vol_ratio_20=vol_ratio_20,
                news_count=len(news_items),
                caution_news_count=caution_news_count,
                flags=flags,
            )

            rows.append(
                {
                    "id": uuid4(),
                    "symbol_id": symbol_id,
                    "report_date": resolved_report_date,
                    "report_type": "daily_snapshot",
                    "status": "ready",
                    "schema_version": "v1",
                    "prompt_version": "",
                    "model_name": None,
                    "summary_text": summary_text,
                    "source_news_ids": [str(n["id"]) for n in news_items if n.get("id")],
                    "feature_snapshot": feature_snapshot,
                    "metrics_payload": metrics_payload,
                    "report_meta": report_meta,
                }
            )

        if rows:
            stmt = pg_insert(report_t).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    report_t.c.symbol_id,
                    report_t.c.report_date,
                    report_t.c.report_type,
                    report_t.c.prompt_version,
                ],
                set_={
                    "status": stmt.excluded.status,
                    "schema_version": stmt.excluded.schema_version,
                    "model_name": stmt.excluded.model_name,
                    "summary_text": stmt.excluded.summary_text,
                    "source_news_ids": stmt.excluded.source_news_ids,
                    "feature_snapshot": stmt.excluded.feature_snapshot,
                    "metrics_payload": stmt.excluded.metrics_payload,
                    "report_meta": stmt.excluded.report_meta,
                },
            )
            result = conn.execute(stmt)
            summary.rows_written = int(result.rowcount or 0)

        summary.processed_symbols = len(rows)
        return summary

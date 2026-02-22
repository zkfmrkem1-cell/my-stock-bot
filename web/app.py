# web/app.py
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import create_db_engine  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
DEFAULT_WATCHLIST = ["QQQM", "QQQ", "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]


def _parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_kst_str(value: Any) -> str:
    if value is None:
        return "-"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _normalize_disparity25_percent(row: pd.Series) -> float | None:
    feature_meta = _parse_jsonish(row.get("feature_meta"))
    close_price = _safe_float(row.get("close_price"))
    ma_25 = _safe_float(feature_meta.get("ma_25"))
    if close_price is not None and ma_25 not in (None, 0):
        return (close_price / ma_25 - 1.0) * 100.0

    stored = _safe_float(feature_meta.get("disparity25"))
    if stored is None:
        return None
    return stored - 100.0 if stored > 50 else stored


@st.cache_resource
def get_engine():
    # Read-only usage: dashboard only runs SELECT queries.
    return create_db_engine(echo=False)


def _require_table(conn: sa.Connection, *, schema: str, table_name: str) -> sa.Table | None:
    inspector = sa.inspect(conn)
    if not inspector.has_table(table_name, schema=schema):
        return None
    return sa.Table(table_name, sa.MetaData(), schema=schema, autoload_with=conn)


@st.cache_data(ttl=300)
def load_health_check() -> tuple[dict[str, Any] | None, pd.DataFrame]:
    engine = get_engine()
    with engine.connect() as conn:
        job_table = _require_table(conn, schema="meta", table_name="job_run")
        if job_table is None:
            return None, pd.DataFrame()

        cols = set(job_table.c.keys())
        selected_names = [c for c in ["id", "job_name", "status", "started_at", "finished_at", "message"] if c in cols]
        stmt = sa.select(*[job_table.c[name] for name in selected_names])
        if "job_name" in cols:
            stmt = stmt.where(job_table.c.job_name == "raw_daily_ingest")
        order_col = job_table.c.started_at if "started_at" in cols else list(job_table.c)[0]
        stmt = stmt.order_by(order_col.desc()).limit(10)
        rows = conn.execute(stmt).mappings().all()

    if not rows:
        return None, pd.DataFrame(columns=selected_names)

    df = pd.DataFrame(rows)
    latest = dict(rows[0])
    return latest, df


@st.cache_data(ttl=300)
def load_latest_ai_report() -> dict[str, Any] | None:
    engine = get_engine()
    with engine.connect() as conn:
        report_table = _require_table(conn, schema="report", table_name="daily_reports")
        if report_table is None:
            return None
        cols = set(report_table.c.keys())
        selected_names = [
            c
            for c in [
                "id",
                "report_date",
                "report_type",
                "title",
                "model_name",
                "summary_text",
                "highlight_symbols",
                "metrics_payload",
                "report_meta",
                "discord_sent",
                "created_at",
            ]
            if c in cols
        ]
        stmt = (
            sa.select(*[report_table.c[name] for name in selected_names])
            .order_by(report_table.c.report_date.desc(), report_table.c.created_at.desc())
            .limit(1)
        )
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


@st.cache_data(ttl=300)
def load_available_metric_tickers() -> tuple[list[str], date | None]:
    engine = get_engine()
    with engine.connect() as conn:
        feat_table = _require_table(conn, schema="feat", table_name="daily_features")
        meta_table = _require_table(conn, schema="meta", table_name="symbols")
        if feat_table is None or meta_table is None:
            return [], None

        latest_trade_date = conn.execute(sa.select(sa.func.max(feat_table.c.trade_date))).scalar_one_or_none()
        if latest_trade_date is None:
            return [], None

        stmt = (
            sa.select(meta_table.c.ticker)
            .select_from(feat_table.join(meta_table, feat_table.c.symbol_id == meta_table.c.id))
            .where(feat_table.c.trade_date == latest_trade_date)
            .distinct()
            .order_by(meta_table.c.ticker.asc())
        )
        tickers = [str(r.ticker).upper() for r in conn.execute(stmt)]
    return tickers, latest_trade_date


@st.cache_data(ttl=300)
def load_metric_table(selected_tickers: tuple[str, ...]) -> tuple[pd.DataFrame, date | None]:
    engine = get_engine()
    with engine.connect() as conn:
        feat_table = _require_table(conn, schema="feat", table_name="daily_features")
        meta_table = _require_table(conn, schema="meta", table_name="symbols")
        if feat_table is None or meta_table is None:
            return pd.DataFrame(), None

        latest_trade_date = conn.execute(sa.select(sa.func.max(feat_table.c.trade_date))).scalar_one_or_none()
        if latest_trade_date is None:
            return pd.DataFrame(), None

        cols = set(feat_table.c.keys())
        select_cols = [
            meta_table.c.ticker.label("ticker"),
            feat_table.c.trade_date.label("trade_date"),
        ]
        for name in ["close_price", "ma_5", "ma_20", "rsi_14", "return_1d", "return_5d", "feature_meta"]:
            if name in cols:
                select_cols.append(feat_table.c[name].label(name))

        stmt = (
            sa.select(*select_cols)
            .select_from(feat_table.join(meta_table, feat_table.c.symbol_id == meta_table.c.id))
            .where(feat_table.c.trade_date == latest_trade_date)
            .order_by(meta_table.c.ticker.asc())
        )
        if selected_tickers:
            stmt = stmt.where(meta_table.c.ticker.in_(list(selected_tickers)))

        rows = conn.execute(stmt).mappings().all()

    if not rows:
        return pd.DataFrame(), latest_trade_date

    df = pd.DataFrame(rows)
    df["disparity25"] = df.apply(_normalize_disparity25_percent, axis=1)

    display_df = pd.DataFrame(
        {
            "Ticker": df["ticker"],
            "Trade Date": df["trade_date"].astype(str),
            "Close": pd.to_numeric(df.get("close_price"), errors="coerce"),
            "Disparity25 (%)": pd.to_numeric(df.get("disparity25"), errors="coerce"),
            "RSI14": pd.to_numeric(df.get("rsi_14"), errors="coerce") if "rsi_14" in df.columns else None,
            "MA5": pd.to_numeric(df.get("ma_5"), errors="coerce") if "ma_5" in df.columns else None,
            "MA20": pd.to_numeric(df.get("ma_20"), errors="coerce") if "ma_20" in df.columns else None,
            "Return 1D (%)": pd.to_numeric(df.get("return_1d"), errors="coerce") * 100 if "return_1d" in df.columns else None,
            "Return 5D (%)": pd.to_numeric(df.get("return_5d"), errors="coerce") * 100 if "return_5d" in df.columns else None,
        }
    )
    if "Disparity25 (%)" in display_df.columns:
        display_df = display_df.sort_values(["Disparity25 (%)", "Ticker"], ascending=[True, True], na_position="last")
    return display_df.reset_index(drop=True), latest_trade_date


def render_health_check() -> None:
    st.subheader("A. 시스템 상태 (Health Check)")
    latest, recent_df = load_health_check()

    if latest is None:
        st.warning("`meta.job_run` 테이블이 없거나 수집 이력이 없습니다.")
        return

    status_raw = str(latest.get("status", "UNKNOWN")).upper()
    if status_raw.startswith("SUCCESS"):
        status_label = "SUCCESS"
        status_color = "green"
    elif "PARTIAL" in status_raw:
        status_label = status_raw
        status_color = "orange"
    elif "RUNNING" in status_raw:
        status_label = status_raw
        status_color = "blue"
    else:
        status_label = status_raw
        status_color = "red"

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Ingest Status", status_label)
    col2.metric("Started (KST)", _to_kst_str(latest.get("started_at")))
    col3.metric("Finished (KST)", _to_kst_str(latest.get("finished_at")))

    message = latest.get("message")
    if message:
        st.caption(f"Message: {message}")

    with st.expander("Recent Job Runs", expanded=False):
        show_df = recent_df.copy()
        for col in ["started_at", "finished_at"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].apply(_to_kst_str)
        st.dataframe(show_df, use_container_width=True, hide_index=True)


def render_latest_ai_report() -> None:
    st.subheader("B. 오늘의 AI 리포트")
    row = load_latest_ai_report()
    if row is None:
        st.warning("`report.daily_reports`에 저장된 리포트가 없습니다.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Report Date", str(row.get("report_date", "-")))
    c2.metric("Report Type", str(row.get("report_type", "-")))
    c3.metric("Discord Sent", "Yes" if bool(row.get("discord_sent")) else "No")

    title = row.get("title")
    if title:
        st.markdown(f"### {title}")

    meta_line = []
    if row.get("created_at") is not None:
        meta_line.append(f"Created: {_to_kst_str(row.get('created_at'))}")
    if row.get("model_name"):
        meta_line.append(f"Model: {row.get('model_name')}")
    if meta_line:
        st.caption(" | ".join(meta_line))

    st.markdown(str(row.get("summary_text", "")))

    with st.expander("Report Metadata", expanded=False):
        st.json(
            {
                "highlight_symbols": row.get("highlight_symbols"),
                "metrics_payload": row.get("metrics_payload"),
                "report_meta": row.get("report_meta"),
            }
        )


def render_metrics_table() -> None:
    st.subheader("C. 주요 지표 테이블")
    available_tickers, latest_trade_date = load_available_metric_tickers()

    if latest_trade_date is None:
        st.warning("`feat.daily_features`에 데이터가 없습니다.")
        return

    default_selection = [t for t in DEFAULT_WATCHLIST if t in available_tickers]
    if not default_selection:
        default_selection = available_tickers[:10]

    selected_tickers = st.multiselect(
        "종목 선택 (최근 영업일 기준)",
        options=available_tickers,
        default=default_selection,
        help="선택하지 않으면 최근 영업일 전체 종목을 표시합니다.",
    )

    metric_df, _ = load_metric_table(tuple(selected_tickers))
    st.caption(f"Latest Trade Date: {latest_trade_date}")

    if metric_df.empty:
        st.info("선택 조건에 해당하는 지표 데이터가 없습니다.")
        return

    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn(format="%.2f"),
            "Disparity25 (%)": st.column_config.NumberColumn(format="%.2f"),
            "RSI14": st.column_config.NumberColumn(format="%.2f"),
            "MA5": st.column_config.NumberColumn(format="%.2f"),
            "MA20": st.column_config.NumberColumn(format="%.2f"),
            "Return 1D (%)": st.column_config.NumberColumn(format="%.2f"),
            "Return 5D (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="Stock AI Dashboard", page_icon="📈", layout="wide")
    st.title("주식 매매 AI 운영 대시보드")
    st.caption("Cloud PostgreSQL read-only monitoring view (Streamlit)")

    try:
        render_health_check()
        st.divider()
        render_latest_ai_report()
        st.divider()
        render_metrics_table()
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        st.error("대시보드 데이터를 불러오는 중 오류가 발생했습니다. 환경변수와 DB 연결 상태를 확인하세요.")


if __name__ == "__main__":
    main()


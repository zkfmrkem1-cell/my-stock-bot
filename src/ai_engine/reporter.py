from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from ..database import _load_dotenv_if_available, create_db_engine

RawMode = Literal["modern_symbol_id", "legacy_ticker"]
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


@dataclass(slots=True)
class AIReportRunResult:
    report_id: str | None
    report_date: date
    report_type: str
    candidate_count: int
    excel_path: str
    discord_sent: bool
    gemini_used: bool
    gemini_model: str | None
    highlight_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


def _require_table(conn: sa.Connection, *, schema: str, table_name: str) -> sa.Table:
    insp = sa.inspect(conn)
    if not insp.has_table(table_name, schema=schema):
        raise RuntimeError(f"Missing required table {schema}.{table_name}.")
    return _reflect_table(conn, schema=schema, table_name=table_name)


def _detect_raw_mode(raw_table: sa.Table) -> RawMode:
    cols = set(raw_table.c.keys())
    if {"symbol_id", "trade_date"}.issubset(cols):
        return "modern_symbol_id"
    if {"ticker", "trade_date"}.issubset(cols):
        return "legacy_ticker"
    raise RuntimeError("raw.ohlcv_daily must include either (symbol_id, trade_date) or (ticker, trade_date).")


def _as_float(value: Any) -> float | None:
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


def _parse_feature_meta(value: Any) -> dict[str, Any]:
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


def _load_latest_trade_date(
    conn: sa.Connection,
    *,
    feat_table: sa.Table,
    meta_table: sa.Table,
    tickers: Sequence[str] | None,
) -> date:
    stmt = sa.select(sa.func.max(feat_table.c.trade_date)).select_from(
        feat_table.join(meta_table, feat_table.c.symbol_id == meta_table.c.id)
    )
    normalized = _normalize_tickers(tickers)
    if normalized:
        stmt = stmt.where(meta_table.c.ticker.in_(normalized))
    latest = conn.execute(stmt).scalar_one_or_none()
    if latest is None:
        raise RuntimeError("No rows found in feat.daily_features for the requested scope.")
    return latest


def _load_latest_joined_candidates_frame(
    conn: sa.Connection,
    *,
    tickers: Sequence[str] | None,
) -> tuple[pd.DataFrame, date]:
    feat_table = _require_table(conn, schema="feat", table_name="daily_features")
    meta_table = _require_table(conn, schema="meta", table_name="symbols")
    raw_table = _require_table(conn, schema="raw", table_name="ohlcv_daily")
    raw_mode = _detect_raw_mode(raw_table)
    latest_trade_date = _load_latest_trade_date(
        conn,
        feat_table=feat_table,
        meta_table=meta_table,
        tickers=tickers,
    )

    feat_meta_col = feat_table.c.feature_meta if "feature_meta" in feat_table.c else None
    select_cols: list[sa.ColumnElement[Any]] = [
        meta_table.c.ticker.label("ticker"),
        feat_table.c.symbol_id.label("symbol_id"),
        feat_table.c.trade_date.label("trade_date"),
    ]
    for col_name in ("close_price", "ma_5", "ma_20", "rsi_14"):
        if col_name in feat_table.c:
            select_cols.append(feat_table.c[col_name].label(col_name))
    if feat_meta_col is not None:
        select_cols.append(feat_meta_col.label("feature_meta"))
    if "close" in raw_table.c:
        select_cols.append(raw_table.c.close.label("raw_close"))
    if "volume" in raw_table.c:
        select_cols.append(raw_table.c.volume.label("raw_volume"))
    if "created_at" in raw_table.c:
        select_cols.append(raw_table.c.created_at.label("raw_created_at"))

    base = feat_table.join(meta_table, feat_table.c.symbol_id == meta_table.c.id)
    if raw_mode == "modern_symbol_id":
        raw_join = sa.and_(
            raw_table.c.symbol_id == feat_table.c.symbol_id,
            raw_table.c.trade_date == feat_table.c.trade_date,
        )
    else:
        raw_join = sa.and_(
            raw_table.c.ticker == meta_table.c.ticker,
            raw_table.c.trade_date == feat_table.c.trade_date,
        )
    base = base.outerjoin(raw_table, raw_join)

    stmt = sa.select(*select_cols).select_from(base).where(feat_table.c.trade_date == latest_trade_date)
    normalized = _normalize_tickers(tickers)
    if normalized:
        stmt = stmt.where(meta_table.c.ticker.in_(normalized))

    rows = conn.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(), latest_trade_date

    df = pd.DataFrame(rows)
    if "raw_created_at" in df.columns:
        df = df.sort_values(["ticker", "trade_date", "raw_created_at"]).drop_duplicates(
            subset=["ticker", "trade_date"], keep="last"
        )
    else:
        df = df.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
    df = df.reset_index(drop=True)
    return df, latest_trade_date


def _compute_disparity25_pct(row: pd.Series) -> float | None:
    feature_meta = _parse_feature_meta(row.get("feature_meta"))
    ma_25 = _as_float(feature_meta.get("ma_25"))
    close_price = _as_float(row.get("close_price"))
    if close_price is not None and ma_25 not in (None, 0):
        return (close_price / ma_25 - 1.0) * 100.0

    stored = _as_float(feature_meta.get("disparity25"))
    if stored is None:
        return None
    # Backward-compat: early processor stored ratio*100 (e.g., 97.3 instead of -2.7).
    return stored - 100.0 if stored > 50 else stored


def extract_bnf_oversold_candidates(
    *,
    tickers: Sequence[str] | None = None,
    disparity_threshold: float = -15.0,
    top_k: int = 5,
    echo: bool = False,
    engine: Engine | None = None,
) -> tuple[pd.DataFrame, date, list[str]]:
    db_engine = engine or create_db_engine(echo=echo)
    warnings: list[str] = []
    with db_engine.connect() as conn:
        joined_df, latest_trade_date = _load_latest_joined_candidates_frame(conn, tickers=tickers)

    if joined_df.empty:
        return pd.DataFrame(), latest_trade_date, warnings

    if "feature_meta" not in joined_df.columns:
        warnings.append("feat.daily_features.feature_meta is missing; disparity25 extraction is limited.")
        joined_df["feature_meta"] = [{} for _ in range(len(joined_df))]

    joined_df["disparity25"] = joined_df.apply(_compute_disparity25_pct, axis=1)
    if "raw_volume" in joined_df.columns:
        joined_df["raw_volume"] = pd.to_numeric(joined_df["raw_volume"], errors="coerce")

    if joined_df["disparity25"].isna().all():
        warnings.append("No valid disparity25 values found in feature_meta.")

    candidates = joined_df[joined_df["disparity25"].astype(float) < float(disparity_threshold)].copy()
    if "rsi_14" in candidates.columns:
        candidates["rsi_14"] = pd.to_numeric(candidates["rsi_14"], errors="coerce")
    sort_cols = ["disparity25"] + (["rsi_14"] if "rsi_14" in candidates.columns else [])
    candidates = candidates.sort_values(sort_cols, ascending=True).head(int(top_k)).reset_index(drop=True)

    preferred_cols = [
        "ticker",
        "trade_date",
        "disparity25",
        "rsi_14",
        "close_price",
        "ma_5",
        "ma_20",
        "raw_close",
        "raw_volume",
        "symbol_id",
    ]
    existing_cols = [c for c in preferred_cols if c in candidates.columns]
    if existing_cols:
        candidates = candidates[existing_cols]
    return candidates, latest_trade_date, warnings


def export_candidates_to_excel(
    *,
    candidates_df: pd.DataFrame,
    analysis_date: date,
    output_dir: str | os.PathLike[str] = "output",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    file_path = out_dir / f"{today_str}_bnf_oversold_candidates.xlsx"

    export_df = candidates_df.copy()
    if "trade_date" in export_df.columns:
        export_df["trade_date"] = export_df["trade_date"].astype(str)

    summary_df = pd.DataFrame(
        [
            {
                "generated_on": today_str,
                "analysis_trade_date": analysis_date.isoformat(),
                "candidate_count": int(len(export_df)),
            }
        ]
    )

    try:
        with pd.ExcelWriter(file_path) as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            export_df.to_excel(writer, sheet_name="candidates", index=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Excel export requires an engine such as openpyxl. Install openpyxl.") from exc

    return file_path


def _serialize_candidates_for_prompt(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    temp = df.copy()
    for col in temp.columns:
        if pd.api.types.is_datetime64_any_dtype(temp[col]):
            temp[col] = temp[col].astype(str)
    if "trade_date" in temp.columns:
        temp["trade_date"] = temp["trade_date"].astype(str)
    # Convert NaN to None for JSON serialization.
    temp = temp.astype(object).where(pd.notna(temp), None)
    return temp.to_dict(orient="records")


def build_bnf_oversold_prompt(
    *,
    analysis_date: date,
    candidates_df: pd.DataFrame,
    disparity_threshold: float,
) -> str:
    data_payload = _serialize_candidates_for_prompt(candidates_df)
    data_text = json.dumps(data_payload, ensure_ascii=False, indent=2)
    return (
        "당신은 퀀트 트레이딩 전문가입니다.\n"
        f"분석일자: {analysis_date.isoformat()}\n"
        f"전략 조건: disparity25 < {disparity_threshold} (BNF 과매도 전략)\n"
        f"과매도 후보 종목(최대 {len(data_payload)}개): {data_text}\n\n"
        "요청사항:\n"
        "1) 각 종목의 기술적 과매도 상태를 요약\n"
        "2) 단기 반등 가능성과 실패 리스크를 균형 있게 설명\n"
        "3) 공통 리스크 요인(시장/섹터/변동성)을 정리\n"
        "4) 마지막에 체크리스트 형태의 관찰 포인트 제시\n"
        "답변은 한국어로 작성하세요."
    )


def _extract_gemini_text(response_json: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response_json.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                texts.append(str(text))
    combined = "\n".join(t.strip() for t in texts if t and str(t).strip()).strip()
    if combined:
        return combined
    prompt_feedback = response_json.get("promptFeedback")
    raise RuntimeError(f"Gemini returned no text. promptFeedback={prompt_feedback}")


def generate_gemini_report(
    *,
    prompt: str,
    gemini_model: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> tuple[str, str]:
    _load_dotenv_if_available()
    key = (api_key or os.getenv(GEMINI_API_KEY_ENV, "")).strip()
    if not key:
        raise RuntimeError(f"Missing required environment variable: {GEMINI_API_KEY_ENV}")

    model = (gemini_model or os.getenv("GEMINI_MODEL", "")).strip() or "gemini-1.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API failed ({resp.status_code}): {resp.text[:1000]}")
    data = resp.json()
    return _extract_gemini_text(data), model


def _build_no_candidate_report(analysis_date: date, disparity_threshold: float) -> str:
    return (
        f"[AI Report] {analysis_date.isoformat()} 기준 BNF 과매도 스캔 결과\n\n"
        f"- 조건: disparity25 < {disparity_threshold}\n"
        "- 결과: 조건을 만족하는 후보 종목이 없습니다.\n"
        "- 해석: 단기 과매도 신호가 충분히 누적된 종목이 없거나, 최근 반등으로 이격도 조건이 완화된 상태일 수 있습니다.\n"
        "- 액션: 조건 완화(예: -12~-10) 또는 관찰 리스트 중심의 모니터링을 검토하세요."
    )


def _clean_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _upsert_daily_report(
    *,
    engine: Engine,
    report_date: date,
    report_type: str,
    title: str,
    summary_text: str,
    highlight_symbols: list[str],
    metrics_payload: dict[str, Any],
    report_meta: dict[str, Any],
    model_name: str | None,
    discord_sent: bool = False,
) -> str | None:
    with engine.begin() as conn:
        report_table = _require_table(conn, schema="report", table_name="daily_reports")
        cols = set(report_table.c.keys())

        payload: dict[str, Any] = {}
        if "id" in cols:
            payload["id"] = uuid4()
        if "report_date" in cols:
            payload["report_date"] = report_date
        if "report_type" in cols:
            payload["report_type"] = report_type
        if "title" in cols:
            payload["title"] = title
        if "model_name" in cols:
            payload["model_name"] = model_name
        if "summary_text" in cols:
            payload["summary_text"] = summary_text
        if "highlight_symbols" in cols:
            payload["highlight_symbols"] = highlight_symbols
        if "metrics_payload" in cols:
            payload["metrics_payload"] = metrics_payload
        if "report_meta" in cols:
            payload["report_meta"] = report_meta
        if "discord_sent" in cols:
            payload["discord_sent"] = bool(discord_sent)

        stmt = pg_insert(report_table).values(payload)
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in report_table.columns
            if c.name not in {"id", "created_at", "report_date", "report_type"}
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[report_table.c.report_date, report_table.c.report_type],
            set_=update_cols,
        )
        if "id" in cols:
            inserted_id = conn.execute(stmt.returning(report_table.c.id)).scalar_one()
            return str(inserted_id)
        conn.execute(stmt)
        return None


def mark_report_discord_sent(
    *,
    engine: Engine,
    report_date: date,
    report_type: str,
) -> None:
    with engine.begin() as conn:
        report_table = _require_table(conn, schema="report", table_name="daily_reports")
        if "discord_sent" not in report_table.c:
            return
        conn.execute(
            sa.update(report_table)
            .where(report_table.c.report_date == report_date)
            .where(report_table.c.report_type == report_type)
            .values(discord_sent=True)
        )


def run_ai_report_pipeline(
    *,
    tickers: Sequence[str] | None = None,
    disparity_threshold: float = -15.0,
    top_k: int = 5,
    skip_gemini: bool = False,
    skip_discord: bool = False,
    gemini_model: str | None = None,
    echo: bool = False,
) -> AIReportRunResult:
    db_engine = create_db_engine(echo=echo)
    warnings: list[str] = []

    candidates_df, analysis_date, extract_warnings = extract_bnf_oversold_candidates(
        tickers=tickers,
        disparity_threshold=disparity_threshold,
        top_k=top_k,
        echo=echo,
        engine=db_engine,
    )
    warnings.extend(extract_warnings)

    excel_path = export_candidates_to_excel(
        candidates_df=candidates_df,
        analysis_date=analysis_date,
        output_dir="output",
    )

    if candidates_df.empty or skip_gemini:
        report_text = _build_no_candidate_report(analysis_date, disparity_threshold) if candidates_df.empty else (
            f"[Mock AI Report]\n분석일자: {analysis_date.isoformat()}\n"
            f"후보 종목 수: {len(candidates_df)}\n"
            "Gemini 호출이 skip 되어 로컬 요약만 저장합니다."
        )
        used_model = None if skip_gemini else gemini_model
        gemini_used = False
        if skip_gemini:
            warnings.append("Gemini call skipped by CLI option.")
        if candidates_df.empty:
            warnings.append("No BNF oversold candidates found for the latest trading day.")
    else:
        prompt = build_bnf_oversold_prompt(
            analysis_date=analysis_date,
            candidates_df=candidates_df,
            disparity_threshold=disparity_threshold,
        )
        report_text, used_model = generate_gemini_report(
            prompt=prompt,
            gemini_model=gemini_model,
        )
        gemini_used = True

    highlight_symbols = candidates_df["ticker"].astype(str).tolist() if not candidates_df.empty else []
    metrics_payload = {
        "strategy": "bnf_oversold_disparity25",
        "threshold_disparity25": float(disparity_threshold),
        "candidate_count": int(len(candidates_df)),
        "top_k": int(top_k),
        "analysis_trade_date": analysis_date.isoformat(),
        "min_disparity25": _clean_jsonable(candidates_df["disparity25"].min()) if "disparity25" in candidates_df else None,
        "max_disparity25": _clean_jsonable(candidates_df["disparity25"].max()) if "disparity25" in candidates_df else None,
    }
    report_meta = {
        "generated_at": _utcnow().isoformat(),
        "excel_path": str(excel_path),
        "source_tables": ["raw.ohlcv_daily", "feat.daily_features", "meta.symbols"],
        "candidate_preview": [
            {k: _clean_jsonable(v) for k, v in row.items()}
            for row in _serialize_candidates_for_prompt(candidates_df)[:5]
        ],
        "warnings": warnings,
    }
    report_type = "bnf_oversold_daily"
    report_title = f"BNF Oversold Report ({analysis_date.isoformat()})"
    report_id = _upsert_daily_report(
        engine=db_engine,
        report_date=analysis_date,
        report_type=report_type,
        title=report_title,
        summary_text=report_text,
        highlight_symbols=highlight_symbols,
        metrics_payload=metrics_payload,
        report_meta=report_meta,
        model_name=used_model,
        discord_sent=False,
    )

    discord_sent = False
    if not skip_discord:
        from .discord_bot import send_discord_report

        send_discord_report(
            report_text=report_text,
            excel_path=excel_path,
        )
        mark_report_discord_sent(
            engine=db_engine,
            report_date=analysis_date,
            report_type=report_type,
        )
        discord_sent = True
    else:
        warnings.append("Discord send skipped by CLI option.")

    return AIReportRunResult(
        report_id=report_id,
        report_date=analysis_date,
        report_type=report_type,
        candidate_count=int(len(candidates_df)),
        excel_path=str(excel_path),
        discord_sent=discord_sent,
        gemini_used=gemini_used,
        gemini_model=used_model,
        highlight_symbols=highlight_symbols,
        warnings=warnings,
    )

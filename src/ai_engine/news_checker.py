"""
뉴스 수집 → DB 저장 모듈
- yfinance + Google News RSS로 뉴스 수집
- news.stock_news 테이블에 저장
- 판단(is_tradeable)은 매매 프로그램이 직접 수행
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import requests
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from ..database import _load_dotenv_if_available, create_db_engine
from ..ticker_aliases import canonicalize_ticker_for_yfinance

_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_MAX_NEWS_PER_SOURCE = 10  # 소스당 최대 수집 뉴스 수


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).casefold()


@dataclass(slots=True)
class NewsCollectResult:
    ticker: str
    symbol_id: int
    news_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NewsDedupeSummary:
    market: str
    dry_run: bool
    duplicate_rows_found: int = 0
    deleted_rows: int = 0
    duplicate_url_rows: int = 0
    duplicate_title_rows: int = 0
    remaining_rows: int = 0


# ---------------------------------------------------------------------------
# 뉴스 수집
# ---------------------------------------------------------------------------

def _fetch_yfinance_news(ticker: str) -> list[dict[str, Any]]:
    """yfinance로 종목 뉴스 수집"""
    try:
        import yfinance as yf
        stock = yf.Ticker(canonicalize_ticker_for_yfinance(ticker))
        news = stock.news or []
        results = []
        for item in news[:_MAX_NEWS_PER_SOURCE]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0), tz=timezone.utc
                ) if item.get("providerPublishTime") else None,
                "source": "yfinance",
            })
        return results
    except Exception:
        return []


def _fetch_google_news_rss(ticker: str, company_name: str | None = None) -> list[dict[str, Any]]:
    """Google News RSS로 종목 뉴스 수집"""
    try:
        import xml.etree.ElementTree as ET
        query = company_name if company_name else canonicalize_ticker_for_yfinance(ticker)
        url = _GOOGLE_NEWS_RSS_URL.format(query=requests.utils.quote(f"{query} stock"))
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        results = []
        for item in items[:_MAX_NEWS_PER_SOURCE]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date_str = item.findtext("pubDate", "")
            published_at = None
            if pub_date_str:
                try:
                    from email.utils import parsedate_to_datetime
                    published_at = parsedate_to_datetime(pub_date_str)
                except Exception:
                    pass
            results.append({
                "title": title,
                "url": link,
                "published_at": published_at,
                "source": "google_rss",
            })
        return results
    except Exception:
        return []


def collect_news(ticker: str, company_name: str | None = None) -> list[dict[str, Any]]:
    """yfinance + Google News RSS 통합 수집 (중복 제목 제거)"""
    yf_news = _fetch_yfinance_news(ticker)
    google_news = _fetch_google_news_rss(ticker, company_name)
    seen_titles: set[str] = set()
    combined = []
    for item in yf_news + google_news:
        title = _norm_text(item.get("title"))
        title_key = _norm_key(title)
        if title and title_key not in seen_titles:
            seen_titles.add(title_key)
            combined.append(item)
    return combined


# ---------------------------------------------------------------------------
# DB 저장
# ---------------------------------------------------------------------------

def _save_news_to_db(
    conn: sa.Connection,
    news_table: sa.Table,
    symbol_id: int,
    news_list: list[dict[str, Any]],
) -> int:
    """news.stock_news에 뉴스 저장, 저장된 row 수 반환"""
    if not news_list:
        return 0
    candidate_rows: list[dict[str, Any]] = []
    candidate_keys: list[tuple[str, str, str]] = []
    seen_batch_keys: set[tuple[str, str, str]] = set()

    for item in news_list:
        title = _norm_text(item.get("title"))
        if not title:
            continue
        source = _norm_text(item.get("source")) or "unknown"
        url = _norm_text(item.get("url")) or None

        dedupe_key = (
            ("url", _norm_key(source), _norm_key(url))
            if url
            else ("title", _norm_key(source), _norm_key(title))
        )
        if dedupe_key in seen_batch_keys:
            continue
        seen_batch_keys.add(dedupe_key)
        candidate_keys.append(dedupe_key)

        candidate_rows.append(
            {
                "id": uuid4(),
                "symbol_id": symbol_id,
                "published_at": item.get("published_at"),
                "title": title,
                "source": source,
                "url": url,
                "sentiment_flag": "normal",  # 기본값, 매매 프로그램이 업데이트
                "ai_comment": None,
                "_dedupe_key": dedupe_key,
            }
        )

    if not candidate_rows:
        return 0

    existing_keys: set[tuple[str, str, str]] = set()
    conditions: list[sa.ColumnElement[bool]] = []
    url_keys = sorted({k[2] for k in candidate_keys if k[0] == "url" and k[2]})
    title_keys = sorted({(k[1], k[2]) for k in candidate_keys if k[0] == "title" and k[2]})
    if url_keys:
        conditions.append(sa.func.lower(sa.func.coalesce(news_table.c.url, "")).in_(url_keys))
    if title_keys:
        conditions.append(
            sa.tuple_(
                sa.func.lower(sa.func.coalesce(news_table.c.source, "")),
                sa.func.lower(sa.func.coalesce(news_table.c.title, "")),
            ).in_(title_keys)
        )

    if conditions:
        stmt = (
            sa.select(news_table.c.source, news_table.c.url, news_table.c.title)
            .where(news_table.c.symbol_id == symbol_id)
            .where(sa.or_(*conditions))
        )
        for row in conn.execute(stmt).mappings():
            source_key = _norm_key(row.get("source"))
            url_key = _norm_key(row.get("url"))
            title_key = _norm_key(row.get("title"))
            if url_key:
                existing_keys.add(("url", source_key, url_key))
            if title_key:
                existing_keys.add(("title", source_key, title_key))

    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if row["_dedupe_key"] in existing_keys:
            continue
        row.pop("_dedupe_key", None)
        rows.append(row)

    if rows:
        conn.execute(sa.insert(news_table), rows)
    return len(rows)


def _market_filter_sql_for_news_dedupe(market: str) -> tuple[str, str]:
    scope = str(market or "all").lower()
    if scope == "kr":
        return "JOIN meta.symbols s ON s.id = n.symbol_id", "AND s.ticker LIKE '%.KS'"
    if scope == "us":
        return "JOIN meta.symbols s ON s.id = n.symbol_id", "AND s.ticker NOT LIKE '%.KS'"
    return "", ""


def _count_duplicate_news_rows(
    conn: sa.Connection,
    *,
    use_url_key: bool,
    market: str,
) -> int:
    join_sql, market_clause = _market_filter_sql_for_news_dedupe(market)
    if use_url_key:
        key_partition = "lower(btrim(n.url))"
        key_where = "nullif(btrim(coalesce(n.url, '')), '') IS NOT NULL"
    else:
        key_partition = "lower(btrim(n.title))"
        key_where = (
            "nullif(btrim(coalesce(n.url, '')), '') IS NULL "
            "AND nullif(btrim(coalesce(n.title, '')), '') IS NOT NULL"
        )
    sql = f"""
WITH ranked AS (
    SELECT
        n.id,
        ROW_NUMBER() OVER (
            PARTITION BY n.symbol_id, lower(coalesce(n.source, '')), {key_partition}
            ORDER BY
                COALESCE(n.published_at, n.created_at) DESC NULLS LAST,
                n.created_at DESC NULLS LAST,
                n.id DESC
        ) AS rn
    FROM news.stock_news n
    {join_sql}
    WHERE {key_where}
    {market_clause}
)
SELECT COUNT(*)::bigint AS cnt
FROM ranked
WHERE rn > 1
"""
    value = conn.execute(sa.text(sql)).scalar_one()
    return int(value or 0)


def _delete_duplicate_news_rows(
    conn: sa.Connection,
    *,
    use_url_key: bool,
    market: str,
) -> int:
    join_sql, market_clause = _market_filter_sql_for_news_dedupe(market)
    if use_url_key:
        key_partition = "lower(btrim(n.url))"
        key_where = "nullif(btrim(coalesce(n.url, '')), '') IS NOT NULL"
    else:
        key_partition = "lower(btrim(n.title))"
        key_where = (
            "nullif(btrim(coalesce(n.url, '')), '') IS NULL "
            "AND nullif(btrim(coalesce(n.title, '')), '') IS NOT NULL"
        )
    sql = f"""
WITH ranked AS (
    SELECT
        n.id,
        ROW_NUMBER() OVER (
            PARTITION BY n.symbol_id, lower(coalesce(n.source, '')), {key_partition}
            ORDER BY
                COALESCE(n.published_at, n.created_at) DESC NULLS LAST,
                n.created_at DESC NULLS LAST,
                n.id DESC
        ) AS rn
    FROM news.stock_news n
    {join_sql}
    WHERE {key_where}
    {market_clause}
)
DELETE FROM news.stock_news t
USING ranked r
WHERE t.id = r.id
  AND r.rn > 1
RETURNING t.id
"""
    deleted_ids = conn.execute(sa.text(sql)).scalars().all()
    return len(deleted_ids)


def _count_news_rows(conn: sa.Connection, *, market: str) -> int:
    join_sql, market_clause = _market_filter_sql_for_news_dedupe(market)
    sql = f"""
SELECT COUNT(*)::bigint AS cnt
FROM news.stock_news n
{join_sql}
WHERE 1=1
{market_clause}
"""
    value = conn.execute(sa.text(sql)).scalar_one()
    return int(value or 0)


def dedupe_stock_news_table(
    *,
    market: str = "all",
    dry_run: bool = False,
    echo: bool = False,
    engine: Engine | None = None,
) -> NewsDedupeSummary:
    _load_dotenv_if_available()
    db_engine = engine or create_db_engine(echo=echo)
    market_scope = str(market or "all").lower()

    with db_engine.begin() as conn:
        insp = sa.inspect(conn)
        if not insp.has_table("stock_news", schema="news"):
            raise RuntimeError("news.stock_news table is missing. Run `python -m src.cli init-db` first.")

        dup_url = _count_duplicate_news_rows(conn, use_url_key=True, market=market_scope)
        dup_title = _count_duplicate_news_rows(conn, use_url_key=False, market=market_scope)
        deleted_total = 0
        if not dry_run:
            deleted_total += _delete_duplicate_news_rows(conn, use_url_key=True, market=market_scope)
            deleted_total += _delete_duplicate_news_rows(conn, use_url_key=False, market=market_scope)
        remaining = _count_news_rows(conn, market=market_scope)

    return NewsDedupeSummary(
        market=market_scope,
        dry_run=bool(dry_run),
        duplicate_rows_found=int(dup_url + dup_title),
        deleted_rows=int(deleted_total),
        duplicate_url_rows=int(dup_url),
        duplicate_title_rows=int(dup_title),
        remaining_rows=int(remaining),
    )


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run_news_collect_pipeline(
    candidates: list[dict[str, Any]],  # [{"ticker": "AAPL", "symbol_id": 1, "name": "..."}, ...]
    engine: Engine | None = None,
) -> list[NewsCollectResult]:
    """
    종목 리스트 → 뉴스 수집 → DB 저장
    판단(is_tradeable)은 하지 않음 - 매매 프로그램이 직접 수행
    """
    _load_dotenv_if_available()
    if engine is None:
        engine = create_db_engine()

    results: list[NewsCollectResult] = []

    with engine.connect() as conn:
        insp = sa.inspect(conn)
        if not insp.has_table("stock_news", schema="news"):
            raise RuntimeError("news.stock_news 테이블이 없습니다. init-db를 먼저 실행하세요.")
        news_table = sa.Table("stock_news", sa.MetaData(), schema="news", autoload_with=conn)

    for candidate in candidates:
        ticker = candidate["ticker"]
        symbol_id = candidate["symbol_id"]
        company_name = candidate.get("name")

        print(f"  [{ticker}] 뉴스 수집 중...")
        news_list = collect_news(ticker, company_name)
        time.sleep(0.5)  # RSS 과호출 방지

        with engine.begin() as conn:
            saved = _save_news_to_db(conn, news_table, symbol_id, news_list)
        print(f"  [{ticker}] {saved}개 저장 완료")

        results.append(NewsCollectResult(
            ticker=ticker,
            symbol_id=symbol_id,
            news_count=saved,
        ))

    return results

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
    url_keys = sorted({k[2] for k in candidate_keys if k[0] == "url"})
    title_keys = sorted({(k[1], k[2]) for k in candidate_keys if k[0] == "title"})
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

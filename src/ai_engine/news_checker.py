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

_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_MAX_NEWS_PER_SOURCE = 10  # 소스당 최대 수집 뉴스 수


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
        stock = yf.Ticker(ticker)
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
        query = company_name if company_name else ticker
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
        title = item.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            combined.append(item)
    return combined


# ---------------------------------------------------------------------------
# DB 저장
# ---------------------------------------------------------------------------

def _save_news_to_db(
    conn: sa.Connection,
    symbol_id: int,
    news_list: list[dict[str, Any]],
) -> int:
    """news.stock_news에 뉴스 저장, 저장된 row 수 반환"""
    if not news_list:
        return 0
    news_table = sa.Table("stock_news", sa.MetaData(), schema="news", autoload_with=conn)
    rows = [
        {
            "id": uuid4(),
            "symbol_id": symbol_id,
            "published_at": item.get("published_at"),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "url": item.get("url"),
            "sentiment_flag": "normal",  # 기본값, 매매 프로그램이 업데이트
            "ai_comment": None,
        }
        for item in news_list
        if item.get("title")
    ]
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

    with engine.begin() as conn:
        insp = sa.inspect(conn)
        if not insp.has_table("stock_news", schema="news"):
            raise RuntimeError("news.stock_news 테이블이 없습니다. init-db를 먼저 실행하세요.")

        for candidate in candidates:
            ticker = candidate["ticker"]
            symbol_id = candidate["symbol_id"]
            company_name = candidate.get("name")

            print(f"  [{ticker}] 뉴스 수집 중...")
            news_list = collect_news(ticker, company_name)
            time.sleep(0.5)  # RSS 과호출 방지

            saved = _save_news_to_db(conn, symbol_id, news_list)
            print(f"  [{ticker}] {saved}개 저장 완료")

            results.append(NewsCollectResult(
                ticker=ticker,
                symbol_id=symbol_id,
                news_count=saved,
            ))

    return results

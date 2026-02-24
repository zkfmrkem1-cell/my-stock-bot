from __future__ import annotations

from typing import Sequence

# yfinance uses '-' for certain share-class tickers.
YFINANCE_TICKER_ALIASES: dict[str, str] = {
    "BFB": "BF-B",
    "BRKB": "BRK-B",
}


def _normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


def canonicalize_ticker_for_yfinance(ticker: str) -> str:
    normalized = _normalize_ticker(ticker)
    return YFINANCE_TICKER_ALIASES.get(normalized, normalized)


def canonicalize_ticker_list_for_storage(tickers: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = canonicalize_ticker_for_yfinance(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out

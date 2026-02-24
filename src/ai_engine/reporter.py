"""
Gemini API 호출 유틸리티
- BNF 과매도 분석 제거됨
- 범용 Gemini 호출 함수만 유지
"""
from __future__ import annotations

import os
from typing import Any

import requests

from ..database import _load_dotenv_if_available

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


def generate_gemini_report(
    *,
    prompt: str,
    gemini_model: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> tuple[str, str]:
    """
    Gemini API 호출 → (응답 텍스트, 사용 모델명) 반환
    """
    _load_dotenv_if_available()
    key = api_key or os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Missing required environment variable: {GEMINI_API_KEY_ENV}")

    model = (gemini_model or os.getenv("GEMINI_MODEL", "")).strip() or "gemini-2.0-flash"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
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
    texts: list[str] = []
    for candidate in data.get("candidates", []) or []:
        for part in (candidate.get("content", {}).get("parts", []) or []):
            t = part.get("text", "")
            if t and str(t).strip():
                texts.append(str(t).strip())
    combined = "\n".join(texts).strip()
    if combined:
        return combined, model

    prompt_feedback = data.get("promptFeedback")
    raise RuntimeError(f"Gemini returned no text. promptFeedback={prompt_feedback}")

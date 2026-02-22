from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from ..database import _load_dotenv_if_available

DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
DISCORD_MAX_CONTENT = 1900


@dataclass(slots=True)
class DiscordSendResult:
    webhook_url_used: bool
    message_count: int
    file_attached: bool


def get_discord_webhook_url() -> str:
    _load_dotenv_if_available()
    url = os.getenv(DISCORD_WEBHOOK_ENV, "").strip()
    if not url:
        raise RuntimeError(f"Missing required environment variable: {DISCORD_WEBHOOK_ENV}")
    return url


def _split_message(text: str, *, max_len: int = DISCORD_MAX_CONTENT) -> list[str]:
    text = (text or "").strip()
    if not text:
        return ["(empty report)"]
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= max_len:
            current += line
            continue
        if current:
            chunks.append(current.rstrip())
            current = ""
        while len(line) > max_len:
            chunks.append(line[:max_len].rstrip())
            line = line[max_len:]
        current = line
    if current:
        chunks.append(current.rstrip())
    return [c for c in chunks if c]


def _post_discord_json(*, webhook_url: str, payload: dict, timeout: int) -> None:
    resp = requests.post(webhook_url, json=payload, timeout=timeout)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"Discord webhook failed ({resp.status_code}): {resp.text[:500]}")


def _post_discord_multipart(
    *,
    webhook_url: str,
    payload: dict,
    file_path: Path,
    timeout: int,
) -> None:
    with file_path.open("rb") as fh:
        files = [
            (
                "files[0]",
                (
                    file_path.name,
                    fh,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            )
        ]
        data = {"payload_json": json.dumps(payload, ensure_ascii=False)}
        resp = requests.post(webhook_url, data=data, files=files, timeout=timeout)
    if resp.status_code not in (200, 204):
        raise RuntimeError(f"Discord webhook file upload failed ({resp.status_code}): {resp.text[:500]}")


def send_discord_report(
    *,
    report_text: str,
    excel_path: str | os.PathLike[str] | None = None,
    webhook_url: str | None = None,
    username: str = "Quant AI Reporter",
    timeout: int = 30,
) -> DiscordSendResult:
    url = webhook_url or get_discord_webhook_url()
    chunks = _split_message(report_text)
    file_path = Path(excel_path) if excel_path else None
    file_exists = bool(file_path and file_path.exists())

    sent_count = 0
    for idx, chunk in enumerate(chunks):
        prefix = f"[AI Report {idx + 1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        payload = {
            "username": username,
            "content": (prefix + chunk)[:2000],
        }

        if idx == 0 and file_exists:
            _post_discord_multipart(
                webhook_url=url,
                payload=payload,
                file_path=file_path,
                timeout=timeout,
            )
        else:
            _post_discord_json(
                webhook_url=url,
                payload=payload,
                timeout=timeout,
            )
        sent_count += 1

    return DiscordSendResult(
        webhook_url_used=True,
        message_count=sent_count,
        file_attached=file_exists,
    )


from __future__ import annotations

import argparse
from datetime import date
import sys
from typing import Sequence

from .database import initialize_database


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="CLI entrypoint for the stock trading AI backend.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser(
        "init-db",
        help="Create PostgreSQL schemas and tables.",
    )
    init_db_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    init_db_parser.set_defaults(handler=_handle_init_db)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Incrementally ingest raw daily OHLCV from yfinance, then run QC.",
    )
    ingest_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker list (space or comma separated). Defaults to active symbols in meta.symbols.",
    )
    ingest_parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    ingest_parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD).",
    )
    ingest_parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore MAX(trade_date) incremental cursor and refresh the selected date range.",
    )
    ingest_parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip post-ingestion QC validation.",
    )
    ingest_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    ingest_parser.set_defaults(handler=_handle_ingest)

    process_parser = subparsers.add_parser(
        "process",
        help="Compute features and labels from raw daily OHLCV and load feat/label schemas.",
    )
    process_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker list (space or comma separated). Defaults to available raw symbols.",
    )
    process_parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    process_parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD).",
    )
    process_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    process_parser.set_defaults(handler=_handle_process)

    ai_report_parser = subparsers.add_parser(
        "ai-report",
        help="Extract BNF oversold candidates, generate AI report, save DB record, and send Discord webhook.",
    )
    ai_report_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter (space or comma separated). Defaults to all symbols in latest feat data.",
    )
    ai_report_parser.add_argument(
        "--disparity-threshold",
        type=float,
        default=-15.0,
        help="BNF oversold threshold for disparity25 (default: -15).",
    )
    ai_report_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of candidates to include in the report.",
    )
    ai_report_parser.add_argument(
        "--gemini-model",
        default=None,
        help="Optional Gemini model override (default: env GEMINI_MODEL or gemini-1.5-flash).",
    )
    ai_report_parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Gemini API call and generate a local placeholder report.",
    )
    ai_report_parser.add_argument(
        "--skip-discord",
        action="store_true",
        help="Skip Discord webhook sending (DB save + Excel export still run).",
    )
    ai_report_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    ai_report_parser.set_defaults(handler=_handle_ai_report)

    pipeline_parser = subparsers.add_parser(
        "pipeline-run",
        help="Reserved command for the end-to-end daily pipeline.",
    )
    pipeline_parser.set_defaults(handler=_handle_pipeline_run)

    return parser


def _handle_init_db(args: argparse.Namespace) -> int:
    initialize_database(echo=bool(args.echo))
    print("Database initialization completed (schemas + tables).")
    return 0


def _handle_ingest(args: argparse.Namespace) -> int:
    from .data_engine.ingestor import ingest_raw_daily_ohlcv
    from .data_engine.qc import run_raw_ohlcv_qc

    ingest_summary = ingest_raw_daily_ohlcv(
        tickers=args.tickers,
        full_refresh=bool(args.full_refresh),
        start_date=args.start_date,
        end_date=args.end_date,
        echo=bool(args.echo),
    )

    qc_summary = None
    if not args.skip_qc and (ingest_summary.touched_symbol_ids or ingest_summary.touched_tickers):
        qc_summary = run_raw_ohlcv_qc(
            symbol_ids=ingest_summary.touched_symbol_ids,
            tickers=ingest_summary.touched_tickers,
            start_date=ingest_summary.min_trade_date,
            end_date=ingest_summary.max_trade_date,
            echo=bool(args.echo),
        )

    print(f"ingest.job_run_id={ingest_summary.job_run_id}")
    print(f"ingest.rows_upserted={ingest_summary.total_rows_upserted}")
    print(f"ingest.processed_tickers={ingest_summary.processed_tickers}")
    print(f"ingest.failed_tickers={len(ingest_summary.failed_tickers)}")
    if qc_summary is not None:
        print(f"qc.checked_rows={qc_summary.checked_rows}")
        print(f"qc.invalid_rows={qc_summary.invalid_rows}")
        print(f"qc.missing_sessions_total={qc_summary.missing_sessions_total}")
    if ingest_summary.failed_tickers:
        for failure in ingest_summary.failed_tickers:
            print(f"[WARN] {failure['ticker']}: {failure['error']}", file=sys.stderr)
        return 1
    return 0


def _handle_process(args: argparse.Namespace) -> int:
    from .data_engine.processor import process_feature_and_label_data

    summary = process_feature_and_label_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        echo=bool(args.echo),
    )

    print(f"process.processed_symbols={summary.processed_symbols}")
    print(f"process.raw_rows_read={summary.raw_rows_read}")
    print(f"process.feature_rows_upserted={summary.feature_rows_upserted}")
    print(f"process.label_rows_upserted={summary.label_rows_upserted}")
    if summary.min_trade_date and summary.max_trade_date:
        print(f"process.trade_date_range={summary.min_trade_date}..{summary.max_trade_date}")
    if summary.processed_tickers:
        print(f"process.tickers={','.join(summary.processed_tickers)}")
    for warning in summary.warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    return 0


def _handle_ai_report(args: argparse.Namespace) -> int:
    from .ai_engine.reporter import run_ai_report_pipeline

    result = run_ai_report_pipeline(
        tickers=args.tickers,
        disparity_threshold=float(args.disparity_threshold),
        top_k=int(args.top_k),
        skip_gemini=bool(args.skip_gemini),
        skip_discord=bool(args.skip_discord),
        gemini_model=args.gemini_model,
        echo=bool(args.echo),
    )

    print(f"ai_report.report_id={result.report_id}")
    print(f"ai_report.report_date={result.report_date}")
    print(f"ai_report.report_type={result.report_type}")
    print(f"ai_report.candidate_count={result.candidate_count}")
    print(f"ai_report.excel_path={result.excel_path}")
    print(f"ai_report.gemini_used={result.gemini_used}")
    if result.gemini_model:
        print(f"ai_report.gemini_model={result.gemini_model}")
    print(f"ai_report.discord_sent={result.discord_sent}")
    if result.highlight_symbols:
        print(f"ai_report.tickers={','.join(result.highlight_symbols)}")
    for warning in result.warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    return 0


def _handle_pipeline_run(_: argparse.Namespace) -> int:
    raise NotImplementedError("pipeline-run is planned for a later milestone.")


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

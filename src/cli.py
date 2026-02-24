from __future__ import annotations

import argparse
from datetime import date, timedelta
import sys
from typing import Sequence

from .database import initialize_database
from .seed_universes import ALL_SEED_TICKERS, KR_KOSPI200_TICKERS, US_SEED_TICKERS
from .ticker_aliases import YFINANCE_TICKER_ALIASES, canonicalize_ticker_list_for_storage


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
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Optional market scope filter (us/kr/all). Applied with --tickers if both are provided.",
    )
    ingest_parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    ingest_parser.add_argument(
        "--default-start-date",
        type=_parse_iso_date,
        default=None,
        help=(
            "Fallback start date for tickers with NO existing data (YYYY-MM-DD). "
            "Defaults to 5 years ago. Use e.g. --default-start-date 2016-01-01 for 10 years on local DB."
        ),
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
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Optional market scope filter (us/kr/all). Applied with --tickers if both are provided.",
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
        "--skip-labels",
        action="store_true",
        help="Skip label table generation (saves DB space; labels are not used by the dashboard).",
    )
    process_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    process_parser.set_defaults(handler=_handle_process)

    news_check_parser = subparsers.add_parser(
        "news-check",
        help="Collect news for all active tickers and save to news.stock_news DB table.",
    )
    news_check_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter (space or comma separated). Defaults to all active symbols.",
    )
    news_check_parser.add_argument(
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Optional market scope filter (us/kr/all). Applied with --tickers if both are provided.",
    )
    news_check_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    news_check_parser.set_defaults(handler=_handle_news_check)

    news_dedupe_parser = subparsers.add_parser(
        "news-dedupe",
        help="Delete duplicated rows from news.stock_news (URL-first, fallback source+title).",
    )
    news_dedupe_parser.add_argument(
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Optional market scope filter (us/kr/all) for dedupe/count.",
    )
    news_dedupe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count duplicates only; do not delete rows.",
    )
    news_dedupe_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    news_dedupe_parser.set_defaults(handler=_handle_news_dedupe)

    symbol_reports_parser = subparsers.add_parser(
        "symbol-reports",
        help="Build per-symbol structured snapshot reports and save to report.symbol_reports.",
    )
    symbol_reports_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker filter (space or comma separated). Defaults to all active symbols with features on report-date.",
    )
    symbol_reports_parser.add_argument(
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Optional market scope filter (us/kr/all). Applied with --tickers if both are provided.",
    )
    symbol_reports_parser.add_argument(
        "--report-date",
        type=_parse_iso_date,
        default=None,
        help="Snapshot date (YYYY-MM-DD). Defaults to latest feat.daily_features trade_date.",
    )
    symbol_reports_parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=7,
        help="News lookback window in days for source_news_ids/metrics (default: 7).",
    )
    symbol_reports_parser.add_argument(
        "--max-news-refs",
        type=int,
        default=10,
        help="Max news references stored per symbol report (default: 10).",
    )
    symbol_reports_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    symbol_reports_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    symbol_reports_parser.set_defaults(handler=_handle_symbol_reports)

    status_parser = subparsers.add_parser(
        "status-report",
        help="Send a system status summary to Discord.",
    )
    status_parser.add_argument(
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="Market label/filter for the status report. Use us/kr for split operations; default: all.",
    )
    status_parser.add_argument(
        "--skip-discord",
        action="store_true",
        help="Print report to stdout only, skip Discord sending.",
    )
    status_parser.set_defaults(handler=_handle_status_report)

    pipeline_parser = subparsers.add_parser(
        "pipeline-run",
        help="Reserved command for the end-to-end daily pipeline.",
    )
    pipeline_parser.set_defaults(handler=_handle_pipeline_run)

    seed_parser = subparsers.add_parser(
        "seed-symbols",
        help="Seed meta.symbols with built-in US/KR ticker universes for ingestion/news tracking.",
    )
    seed_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional custom ticker list (space or comma separated). Overrides --market universe.",
    )
    seed_parser.add_argument(
        "--market",
        choices=["us", "kr", "all"],
        default="us",
        help="Which market universe to seed: us (S&P500 + market/sector ETFs), kr (KOSPI200), all (both).",
    )
    seed_parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy SQL echo logging.",
    )
    seed_parser.set_defaults(handler=_handle_seed_symbols)

    return parser


def _handle_init_db(args: argparse.Namespace) -> int:
    initialize_database(echo=bool(args.echo))
    print("Database initialization completed (schemas + tables).")
    return 0


def _handle_ingest(args: argparse.Namespace) -> int:
    from .data_engine.ingestor import ingest_raw_daily_ohlcv
    from .data_engine.qc import run_raw_ohlcv_qc

    default_start = getattr(args, "default_start_date", None) or (date.today() - timedelta(days=5 * 365))

    ingest_summary = ingest_raw_daily_ohlcv(
        tickers=args.tickers,
        market=getattr(args, "market", "all"),
        full_refresh=bool(args.full_refresh),
        start_date=args.start_date,
        end_date=args.end_date,
        default_start_date=default_start,
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
        market=getattr(args, "market", "all"),
        start_date=args.start_date,
        end_date=args.end_date,
        skip_labels=bool(getattr(args, "skip_labels", False)),
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


def _handle_news_check(args: argparse.Namespace) -> int:
    from .ai_engine.news_checker import run_news_collect_pipeline
    from .database import create_db_engine, _load_dotenv_if_available
    import sqlalchemy as sa

    _load_dotenv_if_available()
    engine = create_db_engine(echo=bool(args.echo))

    # 전체 활성 종목 조회
    market_scope = str(getattr(args, "market", "all") or "all").lower()

    with engine.connect() as conn:
        meta_table = sa.Table("symbols", sa.MetaData(), schema="meta", autoload_with=conn)
        stmt = sa.select(meta_table.c.id, meta_table.c.ticker, meta_table.c.name).where(
            meta_table.c.is_active.is_(True)
        )
        if market_scope == "kr":
            stmt = stmt.where(meta_table.c.ticker.like("%.KS"))
        elif market_scope == "us":
            stmt = stmt.where(sa.not_(meta_table.c.ticker.like("%.KS")))
        if args.tickers:
            from .data_engine.ingestor import _normalize_tickers
            tickers_filter = _normalize_tickers(args.tickers)
            stmt = stmt.where(meta_table.c.ticker.in_(tickers_filter))
        rows = conn.execute(stmt).fetchall()

    candidates = [{"symbol_id": r[0], "ticker": r[1], "name": r[2]} for r in rows]
    print(f"news_check.total_tickers={len(candidates)}")

    results = run_news_collect_pipeline(candidates=candidates, engine=engine)

    total_news = sum(r.news_count for r in results)
    print(f"news_check.total_news_saved={total_news}")
    for r in results:
        print(f"  {r.ticker}: {r.news_count}개")
    return 0


def _handle_news_dedupe(args: argparse.Namespace) -> int:
    from .ai_engine.news_checker import dedupe_stock_news_table

    summary = dedupe_stock_news_table(
        market=getattr(args, "market", "all"),
        dry_run=bool(getattr(args, "dry_run", False)),
        echo=bool(getattr(args, "echo", False)),
    )

    print(f"news_dedupe.market={summary.market}")
    print(f"news_dedupe.dry_run={str(summary.dry_run).lower()}")
    print(f"news_dedupe.duplicate_rows_found={summary.duplicate_rows_found}")
    print(f"news_dedupe.duplicate_url_rows={summary.duplicate_url_rows}")
    print(f"news_dedupe.duplicate_title_rows={summary.duplicate_title_rows}")
    print(f"news_dedupe.deleted_rows={summary.deleted_rows}")
    print(f"news_dedupe.remaining_rows={summary.remaining_rows}")
    return 0


def _handle_symbol_reports(args: argparse.Namespace) -> int:
    from .ai_engine.symbol_report_builder import build_symbol_reports
    from .data_engine.ingestor import _normalize_tickers

    # Ensure new report tables exist without requiring a separate manual init-db step.
    initialize_database(echo=bool(args.echo))

    tickers = _normalize_tickers(args.tickers) if args.tickers else None
    summary = build_symbol_reports(
        report_date=args.report_date,
        tickers=tickers,
        market=getattr(args, "market", "all"),
        news_lookback_days=max(1, int(getattr(args, "news_lookback_days", 7) or 7)),
        max_news_refs_per_symbol=max(1, int(getattr(args, "max_news_refs", 10) or 10)),
        limit=args.limit,
        echo=bool(args.echo),
    )

    print(f"symbol_reports.report_date={summary.report_date}")
    print(f"symbol_reports.processed_symbols={summary.processed_symbols}")
    print(f"symbol_reports.rows_written={summary.rows_written}")
    for warning in summary.warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    return 0


def _handle_status_report(args: argparse.Namespace) -> int:
    from .database import create_db_engine, _load_dotenv_if_available
    from datetime import datetime, timezone, timedelta
    import sqlalchemy as sa

    _load_dotenv_if_available()
    engine = create_db_engine()
    market_scope = str(getattr(args, "market", "all") or "all").lower()

    def _meta_market_filter(meta_table: sa.Table):
        if market_scope == "kr":
            return meta_table.c.ticker.like("%.KS")
        if market_scope == "us":
            return sa.not_(meta_table.c.ticker.like("%.KS"))
        return None

    def _apply_optional_where(stmt, condition):
        return stmt.where(condition) if condition is not None else stmt

    def _count_table(
        conn: sa.Connection,
        *,
        schema: str,
        table: str,
        meta_t: sa.Table,
    ) -> int | None:
        insp = sa.inspect(conn)
        if not insp.has_table(table, schema=schema):
            return None
        t = sa.Table(table, sa.MetaData(), schema=schema, autoload_with=conn)
        condition = _meta_market_filter(meta_t)
        if market_scope == "all" or table == "job_run":
            return conn.execute(sa.select(sa.func.count()).select_from(t)).scalar()

        if schema == "meta" and table == "symbols":
            stmt = sa.select(sa.func.count()).select_from(meta_t)
            stmt = _apply_optional_where(stmt, _meta_market_filter(meta_t))
            return conn.execute(stmt).scalar()

        if schema == "raw" and table == "ohlcv_daily":
            if "symbol_id" in t.c:
                stmt = sa.select(sa.func.count()).select_from(t.join(meta_t, meta_t.c.id == t.c.symbol_id))
            elif "ticker" in t.c:
                stmt = sa.select(sa.func.count()).select_from(
                    t.join(meta_t, sa.func.upper(meta_t.c.ticker) == sa.func.upper(t.c.ticker))
                )
            else:
                stmt = sa.select(sa.func.count()).select_from(t)
            stmt = _apply_optional_where(stmt, condition)
            return conn.execute(stmt).scalar()

        if "symbol_id" in t.c:
            stmt = sa.select(sa.func.count()).select_from(t.join(meta_t, meta_t.c.id == t.c.symbol_id))
            stmt = _apply_optional_where(stmt, condition)
            return conn.execute(stmt).scalar()

        return conn.execute(sa.select(sa.func.count()).select_from(t)).scalar()

    with engine.connect() as conn:
        insp = sa.inspect(conn)
        meta_t = sa.Table("symbols", sa.MetaData(), schema="meta", autoload_with=conn)
        market_filter = _meta_market_filter(meta_t)
        # DB 현황
        tables = [
            ("meta", "symbols"),
            ("raw", "ohlcv_daily"),
            ("feat", "daily_features"),
            ("news", "stock_news"),
            ("report", "symbol_reports"),
        ]
        counts = {}
        for schema, table in tables:
            key = f"{schema}.{table}"
            counts[key] = _count_table(conn, schema=schema, table=table, meta_t=meta_t)

        raw_t = sa.Table("ohlcv_daily", sa.MetaData(), schema="raw", autoload_with=conn)
        if market_scope == "all":
            raw_min_date, raw_max_date = conn.execute(
                sa.select(sa.func.min(raw_t.c.trade_date), sa.func.max(raw_t.c.trade_date))
            ).one()
        else:
            if "symbol_id" in raw_t.c:
                raw_from = raw_t.join(meta_t, meta_t.c.id == raw_t.c.symbol_id)
            else:
                raw_from = raw_t.join(meta_t, sa.func.upper(meta_t.c.ticker) == sa.func.upper(raw_t.c.ticker))
            stmt = sa.select(sa.func.min(raw_t.c.trade_date), sa.func.max(raw_t.c.trade_date)).select_from(raw_from)
            stmt = _apply_optional_where(stmt, market_filter)
            raw_min_date, raw_max_date = conn.execute(stmt).one()

        # 최근 수집 뉴스 샘플 (최근 24시간)
        news_t = sa.Table("stock_news", sa.MetaData(), schema="news", autoload_with=conn)
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        sample_stmt = (
            sa.select(meta_t.c.ticker, news_t.c.title, news_t.c.source)
            .join(meta_t, meta_t.c.id == news_t.c.symbol_id)
            .where(news_t.c.created_at >= since)
            .order_by(news_t.c.created_at.desc())
            .limit(10)
        )
        if market_filter is not None:
            sample_stmt = sample_stmt.where(market_filter)
        sample = conn.execute(sample_stmt).fetchall()

        # 최근 ingest 실행 정보
        job_t = sa.Table("job_run", sa.MetaData(), schema="meta", autoload_with=conn)
        last_job = conn.execute(
            sa.select(job_t.c.job_name, job_t.c.status, job_t.c.finished_at)
            .order_by(job_t.c.started_at.desc())
            .limit(3)
        ).fetchall()

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    market_tag = market_scope.upper() if market_scope != "all" else "ALL"
    market_name_ko = {"all": "전체", "us": "미국", "kr": "국내"}.get(market_scope, market_scope.upper())
    lines = [
        f"[시스템 현황][{market_tag}] {today}",
        "",
        "== DB 현황 ==",
        f"- 종목 수: {counts['meta.symbols']:,}개",
        (
            f"- OHLCV 데이터: {counts['raw.ohlcv_daily']:,}건 "
            f"({raw_min_date} ~ {raw_max_date})"
            if raw_min_date and raw_max_date
            else f"- OHLCV 데이터: {counts['raw.ohlcv_daily']:,}건"
        ),
        f"- 피처 데이터: {counts['feat.daily_features']:,}건 (disparity25, RSI, MA)",
        f"- 뉴스 데이터: {counts['news.stock_news']:,}건",
        "",
        "== 최근 작업 ==",
    ]
    if counts.get("report.symbol_reports") is not None:
        lines.insert(6, f"- 종목 리포트 데이터: {counts['report.symbol_reports']:,}건")
    if market_scope != "all":
        lines.insert(3, f"- 리포트 범위: {market_name_ko} 시장")
    for job in last_job:
        finished = job[2].strftime("%m/%d %H:%M") if job[2] else "진행중"
        lines.append(f"- {job[0]}: {job[1]} ({finished})")

    # 뉴스 Gemini 요약 (최근 24시간, 상위 20개 종목 × 3개 헤드라인)
    news_summary = ""
    if sample:
        import json
        from collections import defaultdict
        ticker_map: dict = defaultdict(list)
        for r in sample:
            ticker_map[r[0]].append(r[1])

        try:
            from .ai_engine.reporter import generate_gemini_report
            data_text = json.dumps(dict(ticker_map), ensure_ascii=False)
            market_desc = {"us": "미국", "kr": "한국", "all": "미국/한국"}.get(market_scope, "시장")
            prompt = (
                "당신은 주식 뉴스 분석가입니다.\n"
                f"아래는 오늘 수집된 {market_desc} 주식 종목별 뉴스 헤드라인입니다.\n"
                "각 종목의 핵심 이슈를 한국어로 1줄씩 요약하세요.\n"
                "특이 악재가 있으면 [주의] 태그를 붙이세요.\n"
                "형식: TICKER: 요약내용\n\n"
                f"{data_text}"
            )
            news_summary, _ = generate_gemini_report(prompt=prompt, timeout=45)
        except Exception as e:
            news_summary = f"(요약 실패: {e})"

    if news_summary:
        lines += ["", "== 오늘의 뉴스 요약 ==", news_summary]

    lines += [
        "",
        "== 다음 단계 ==",
        "- 한국투자증권 KIS API 연동",
        "- Oracle Cloud VM 셋업",
        "- GitHub Actions 자동화",
    ]

    report = "\n".join(lines)
    print(report)

    if not args.skip_discord:
        from .ai_engine.discord_bot import send_discord_report
        username = "시스템 리포터" if market_scope == "all" else f"시스템 리포터-{market_tag}"
        result = send_discord_report(report_text=report, username=username)
        print(f"Discord 전송 완료: {result.message_count}개 메시지")

    return 0


def _handle_pipeline_run(_: argparse.Namespace) -> int:
    raise NotImplementedError("pipeline-run is planned for a later milestone.")


# Seed ticker universes are maintained in src/seed_universes.py


def _handle_seed_symbols(args: argparse.Namespace) -> int:
    from .data_engine.ingestor import _normalize_tickers
    from .database import create_db_engine
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    if args.tickers:
        tickers = _normalize_tickers(args.tickers)
    elif args.market == "kr":
        tickers = list(KR_KOSPI200_TICKERS)
    elif args.market == "all":
        tickers = list(ALL_SEED_TICKERS)
    else:
        tickers = list(US_SEED_TICKERS)
    original_ticker_count = len(tickers)
    tickers = canonicalize_ticker_list_for_storage(tickers)
    engine = create_db_engine(echo=bool(args.echo))

    with engine.begin() as conn:
        symbols_table = sa.Table("symbols", sa.MetaData(), schema="meta", autoload_with=conn)
        rows = [{"ticker": t} for t in tickers]
        stmt = (
            pg_insert(symbols_table)
            .values(rows)
            .on_conflict_do_nothing(index_elements=[symbols_table.c.ticker])
            .returning(symbols_table.c.ticker)
        )
        inserted_rows = conn.execute(stmt).scalars().all()
        inserted = len(inserted_rows)

        deactivated_legacy_aliases = 0
        if args.market in {"us", "all"}:
            alias_keys = list(YFINANCE_TICKER_ALIASES.keys())
            if alias_keys:
                result = conn.execute(
                    sa.update(symbols_table)
                    .where(
                        symbols_table.c.ticker.in_(alias_keys),
                        symbols_table.c.is_active.is_(True),
                    )
                    .values(is_active=False)
                )
                deactivated_legacy_aliases = max(int(result.rowcount or 0), 0)

    print(f"seed_symbols.market={args.market}")
    print(f"seed_symbols.total_tickers={len(tickers)}")
    if original_ticker_count != len(tickers):
        print(f"seed_symbols.alias_normalized={original_ticker_count - len(tickers)}")
    if len(tickers) <= 80:
        print(f"seed_symbols.tickers={','.join(tickers)}")
    else:
        preview = ",".join(tickers[:25])
        print(f"seed_symbols.tickers_preview={preview},... ({len(tickers)} total)")
    print(f"seed_symbols.inserted={inserted}")
    print(f"seed_symbols.skipped={len(tickers) - inserted}")
    if args.market in {"us", "all"}:
        print(f"seed_symbols.deactivated_legacy_aliases={deactivated_legacy_aliases}")
    return 0


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

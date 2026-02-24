# GitHub Actions Market Pipeline (Neon + GitHub)

## What Runs Where

- `Neon`: managed PostgreSQL database only
- `GitHub Actions`: scheduled execution for `seed-symbols / ingest / process / news-check / symbol-reports / status-report`

This matches the current architecture: DB in cloud, pipeline compute in cloud runner.

## Workflows Added

- `.github/workflows/pipeline_kr.yml`
- `.github/workflows/pipeline_us.yml`

Both run:

1. `seed-symbols --market <kr|us>`
2. `ingest --market <kr|us> --default-start-date 2016-01-01`
3. `process --market <kr|us> --start-date <today-180d> --skip-labels`
4. `news-check --market <kr|us>`
5. `news-dedupe --market <kr|us>` (automatic duplicate cleanup)
6. `symbol-reports --market <kr|us>`
7. `status-report --market <kr|us>` (Discord send)

## Schedules

- `KR`: `40 6 * * 1-5` (15:40 KST weekdays)
- `US`: `30 22 * * 1-5` (22:30 UTC, after US close year-round)

Note: GitHub cron is UTC only, so the US job time shifts in local ET when DST changes.

## GitHub Secrets / Environments (Recommended)

Create GitHub Environments:

- `kr`
- `us`

Set these secrets in each environment (same names, values can differ by market):

- `POSTGRES_DSN`
- `GEMINI_API_KEY`
- `DISCORD_WEBHOOK_URL`
- `GEMINI_MODEL` (optional, default works if omitted)

Recommended: use separate market env secrets (and usually separate DB/schema targets) for `us` and `kr`.

## Manual Run

Use `Actions` tab and run either workflow manually.

- Optional input: `process_lookback_days` (default `180`)

## Symbol Additions (How "Natural" It Is)

What is automatic now:

- Daily `seed-symbols` runs before ingest
- New tickers from the current built-in universe are inserted automatically
- `ingest --default-start-date 2016-01-01` gives new tickers a 10-year fallback start date

What is not automatic yet:

- `src/seed_universes.py` is a static snapshot (S&P500/KOSPI200 lists)
- Index constituent changes are not fetched from the internet automatically

If you want fully automatic index membership sync, add a separate "universe refresh" job later.

## Notes

- These workflows support market-scoped execution (`--market`) across ingest/process/news-check/symbol-reports/status-report.
- Separate environment secrets / DSNs are still recommended for isolation and simpler operations.
- `status-report` is market-aware (`--market us|kr`) and Discord messages include market labels.
- `meta.job_run` remains a shared log table unless you separate DBs/schemas, so recent-job lines can still look mixed in reports.

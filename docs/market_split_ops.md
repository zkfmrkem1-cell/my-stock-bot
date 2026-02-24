# Market Split Ops (US / KR)

This repo can stay as a single codebase while running separate pipelines for US and KR markets.

## Recommended layout

- One VM / one PostgreSQL instance
- Two databases: `stock_us`, `stock_kr`
- One codebase
- Two runtime configs (`.env.us`, `.env.kr`)
- Two schedulers/jobs

## Env files

Create these files from templates:

- `.ai/.env.shared` from `.ai/.env.shared.example`
- `.ai/.env.us` from `.ai/.env.us.example`
- `.ai/.env.kr` from `.ai/.env.kr.example`

Notes:

- Keep secrets only in real env files, not in `*.example`.
- `POSTGRES_DSN` should point to `stock_us` or `stock_kr`.
- Current Python code reads `POSTGRES_DSN` from environment, so the scripts below set env first and then run CLI.

## Pipeline commands (PowerShell)

US:

```powershell
.\scripts\Run-US-Pipeline.ps1
```

KR:

```powershell
.\scripts\Run-KR-Pipeline.ps1 -SkipNews -SkipStatusReport
```

Examples:

```powershell
.\scripts\Run-US-Pipeline.ps1 -Tickers AAPL MSFT NVDA -StartDate 2026-01-01 -SkipDiscord
.\scripts\Run-US-Pipeline.ps1 -FullRefresh -SkipLabels
```

Generated logs:

- `output/us/ingest.out.log`
- `output/us/process.out.log`
- `output/us/news.out.log`
- `output/us/status_report.out.log`
- `output/kr/...`

## Dashboard commands (PowerShell)

US dashboard:

```powershell
.\scripts\Run-US-Dashboard.ps1
```

KR dashboard (separate port):

```powershell
.\scripts\Run-KR-Dashboard.ps1
```

You can override the port:

```powershell
.\scripts\Run-KR-Dashboard.ps1 -Port 8503
```

## Scheduler naming (recommended)

- `stock-us-pipeline`
- `stock-kr-pipeline`
- `stock-us-dashboard`
- `stock-kr-dashboard`

## Operational rule of thumb

- Analysis/report DBs: `stock_us`, `stock_kr`
- Trading DBs (future): `trade_us`, `trade_kr`


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_market_pipeline.sh --market US|KR [options]

Options:
  --market US|KR            Target market env (.ai/.env.us or .ai/.env.kr)
  --tickers "AAPL MSFT"     Optional tickers (space-separated string)
  --start-date YYYY-MM-DD   Optional start date
  --end-date YYYY-MM-DD     Optional end date
  --full-refresh            Enable full refresh on ingest
  --skip-qc                 Skip ingest QC
  --skip-labels             Skip label generation
  --skip-news               Skip news collection
  --skip-symbol-reports     Skip per-symbol report generation
  --skip-status-report      Skip status report
  --skip-discord            Pass --skip-discord to status-report
EOF
}

MARKET=""
TICKERS_STR=""
START_DATE=""
END_DATE=""
FULL_REFRESH=0
SKIP_QC=0
SKIP_LABELS=0
SKIP_NEWS=0
SKIP_SYMBOL_REPORTS=0
SKIP_STATUS_REPORT=0
SKIP_DISCORD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --market) MARKET="${2:-}"; shift 2 ;;
    --tickers) TICKERS_STR="${2:-}"; shift 2 ;;
    --start-date) START_DATE="${2:-}"; shift 2 ;;
    --end-date) END_DATE="${2:-}"; shift 2 ;;
    --full-refresh) FULL_REFRESH=1; shift ;;
    --skip-qc) SKIP_QC=1; shift ;;
    --skip-labels) SKIP_LABELS=1; shift ;;
    --skip-news) SKIP_NEWS=1; shift ;;
    --skip-symbol-reports) SKIP_SYMBOL_REPORTS=1; shift ;;
    --skip-status-report) SKIP_STATUS_REPORT=1; shift ;;
    --skip-discord) SKIP_DISCORD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MARKET" ]]; then
  echo "--market is required" >&2
  usage
  exit 2
fi

MARKET_UPPER="$(printf '%s' "$MARKET" | tr '[:lower:]' '[:upper:]')"
if [[ "$MARKET_UPPER" != "US" && "$MARKET_UPPER" != "KR" ]]; then
  echo "Invalid --market: $MARKET" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/common/load_dotenv.sh"
import_market_env "$MARKET_UPPER" "$REPO_ROOT"

MARKET_KEY="$(printf '%s' "$MARKET_UPPER" | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$REPO_ROOT/output/$MARKET_KEY"
mkdir -p "$OUTPUT_DIR"

declare -a TICKER_ARGS=()
if [[ -n "$TICKERS_STR" ]]; then
  # shellcheck disable=SC2206
  TICKER_ARR=($TICKERS_STR)
  TICKER_ARGS=(--tickers "${TICKER_ARR[@]}")
fi

invoke_step() {
  local step_name="$1"
  shift
  local out_log="$OUTPUT_DIR/${step_name}.out.log"
  local err_log="$OUTPUT_DIR/${step_name}.err.log"

  printf '\n=== [%s] %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$out_log"
  printf '\n=== [%s] %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$err_log"

  (
    cd "$REPO_ROOT"
    "$@"
  ) >>"$out_log" 2>>"$err_log"
}

declare -a INGEST_CMD=(python -m src.cli ingest --market "$MARKET_KEY" "${TICKER_ARGS[@]}")
[[ -n "$START_DATE" ]] && INGEST_CMD+=(--start-date "$START_DATE")
[[ -n "$END_DATE" ]] && INGEST_CMD+=(--end-date "$END_DATE")
[[ $FULL_REFRESH -eq 1 ]] && INGEST_CMD+=(--full-refresh)
[[ $SKIP_QC -eq 1 ]] && INGEST_CMD+=(--skip-qc)
invoke_step ingest "${INGEST_CMD[@]}"

declare -a PROCESS_CMD=(python -m src.cli process --market "$MARKET_KEY" "${TICKER_ARGS[@]}")
[[ -n "$START_DATE" ]] && PROCESS_CMD+=(--start-date "$START_DATE")
[[ -n "$END_DATE" ]] && PROCESS_CMD+=(--end-date "$END_DATE")
[[ $SKIP_LABELS -eq 1 ]] && PROCESS_CMD+=(--skip-labels)
invoke_step process "${PROCESS_CMD[@]}"

if [[ $SKIP_NEWS -ne 1 ]]; then
  declare -a NEWS_CMD=(python -m src.cli news-check --market "$MARKET_KEY" "${TICKER_ARGS[@]}")
  invoke_step news "${NEWS_CMD[@]}"
fi

if [[ $SKIP_SYMBOL_REPORTS -ne 1 ]]; then
  declare -a SYMBOL_REPORT_CMD=(python -m src.cli symbol-reports --market "$MARKET_KEY" "${TICKER_ARGS[@]}")
  invoke_step symbol_reports "${SYMBOL_REPORT_CMD[@]}"
fi

if [[ $SKIP_STATUS_REPORT -ne 1 ]]; then
  declare -a STATUS_CMD=(python -m src.cli status-report --market "$MARKET_KEY")
  [[ $SKIP_DISCORD -eq 1 ]] && STATUS_CMD+=(--skip-discord)
  invoke_step status_report "${STATUS_CMD[@]}"
fi

echo "[$MARKET_UPPER] Pipeline completed. Logs: $OUTPUT_DIR"

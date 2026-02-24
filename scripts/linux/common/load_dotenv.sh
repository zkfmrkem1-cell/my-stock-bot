#!/usr/bin/env bash
set -euo pipefail

set_dotenv_from_file() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    echo "Env file not found: $file_path" >&2
    return 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue

    if [[ "$line" == export\ * ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi

    if [[ "$line" != *=* ]]; then
      continue
    fi

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ ${#value} -ge 2 ]]; then
      if [[ ( "$value" == \"*\" && "$value" == *\" ) || ( "$value" == \'*\' && "$value" == *\' ) ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    export "$key=$value"
  done < "$file_path"
}

import_market_env() {
  local market="$1"
  local repo_root="$2"
  local market_lc
  market_lc="$(printf '%s' "$market" | tr '[:upper:]' '[:lower:]')"

  local shared_file="$repo_root/.ai/.env.shared"
  local market_file="$repo_root/.ai/.env.${market_lc}"

  if [[ -f "$shared_file" ]]; then
    set_dotenv_from_file "$shared_file"
  fi
  set_dotenv_from_file "$market_file"
}

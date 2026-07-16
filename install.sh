#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" "$ROOT/install.py" "$@"
  fi
done

echo "未找到 Python 3.11、3.12 或 3.13。" >&2
exit 2

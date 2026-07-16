#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

status=0
if command -v python3.13 >/dev/null 2>&1; then
  python3.13 install.py "$@" || status=$?
elif command -v python3.12 >/dev/null 2>&1; then
  python3.12 install.py "$@" || status=$?
elif command -v python3.11 >/dev/null 2>&1; then
  python3.11 install.py "$@" || status=$?
elif command -v python3 >/dev/null 2>&1; then
  python3 install.py "$@" || status=$?
else
  echo "未找到 Python 3.11、3.12 或 3.13。" >&2
  status=2
fi

if [[ -t 0 ]]; then
  echo
  read -r -p "按回车键关闭窗口……" _
fi
exit "$status"

#!/usr/bin/env python3
"""Launch rice_black_bg_finalize.py with the isolated installed runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def standard_runtime_python() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ps-cutout-image" / ".venv" / "Scripts" / "python.exe"
    return Path.home() / ".local" / "share" / "ps-cutout-image" / ".venv" / "bin" / "python"


def configured_python() -> Path | None:
    config_path = ROOT / "runtime.json"
    if config_path.exists():
        try:
            configured = Path(json.loads(config_path.read_text(encoding="utf-8"))["python"])
            if configured.exists():
                return configured
        except (KeyError, OSError, TypeError, ValueError):
            pass

    standard = standard_runtime_python()
    if standard.exists():
        return standard
    return None


def main() -> int:
    python = configured_python()
    if python is None:
        print(
            "尚未找到 PS_cutout image 隔离运行环境。请先运行本目录的 install.py。",
            file=sys.stderr,
        )
        return 2

    script = ROOT / "scripts" / "rice_black_bg_finalize.py"
    if not script.exists():
        print(f"缺少处理脚本：{script}", file=sys.stderr)
        return 2
    if len(sys.argv) == 1:
        subprocess.run([str(python), str(script), "--help"], check=False)
        return 2
    return subprocess.run([str(python), str(script), *sys.argv[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

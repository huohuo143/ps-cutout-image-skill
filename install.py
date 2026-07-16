#!/usr/bin/env python3
"""Install PS_cutout image and its isolated CPU runtime.

This bootstrap uses only the Python standard library. It copies the skill to
the user's global Codex skill directory, creates a dedicated virtual
environment, installs pinned dependencies, downloads the U2Net model, and runs
an end-to-end PSD/PNG self-check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


SKILL_NAME = "ps-cutout-image"
ROOT = Path(__file__).resolve().parent
SUPPORTED_MIN = (3, 11)
SUPPORTED_MAX = (3, 14)


def default_runtime_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ps-cutout-image"
    return Path.home() / ".local" / "share" / "ps-cutout-image"


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str], description: str) -> None:
    print(f"\n==> {description}")
    subprocess.run(command, check=True)


def copy_payload(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for filename in (
        "SKILL.md",
        "README.md",
        "requirements.txt",
        "install.py",
        "install.command",
        "install.sh",
        "install.ps1",
        "run_cutout.py",
        "LICENSE",
    ):
        source = ROOT / filename
        if source.exists():
            destination = target / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    scripts_target = target / "scripts"
    scripts_target.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "scripts").glob("*.py"):
        destination = scripts_target / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="安装 PS_cutout image 技能及隔离运行环境")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.home() / ".agents" / "skills",
        help="Codex 全局技能根目录（默认：~/.agents/skills）",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=default_runtime_dir(),
        help="隔离运行环境目录",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过 U2Net 模型预下载；首次抠图时仍会联网下载",
    )
    parser.add_argument("--dry-run", action="store_true", help="只显示安装计划，不写入文件")
    args = parser.parse_args()

    version = sys.version_info[:2]
    if not (SUPPORTED_MIN <= version < SUPPORTED_MAX):
        print(
            "需要 Python 3.11、3.12 或 3.13。"
            f"当前版本为 {sys.version.split()[0]}。请先安装受支持版本后重试。",
            file=sys.stderr,
        )
        return 2

    target = args.dest.expanduser().resolve() / SKILL_NAME
    runtime_dir = args.runtime_dir.expanduser().resolve()
    environment = runtime_dir / ".venv"
    python = venv_python(environment)

    print("PS_cutout image 安装计划")
    print(f"  源目录：{ROOT}")
    print(f"  技能目录：{target}")
    print(f"  隔离环境：{environment}")
    print(f"  Python：{sys.executable} ({sys.version.split()[0]})")
    print(f"  预下载模型：{'否' if args.skip_model else '是'}")
    if args.dry_run:
        return 0

    runtime_dir.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        print("\n==> 创建隔离 Python 环境")
        venv.EnvBuilder(with_pip=True).create(environment)

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], "更新隔离环境中的 pip")
    run(
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        "安装经过固定版本测试的依赖",
    )

    if not args.skip_model:
        model_probe = (
            "from rembg import new_session; "
            "s=new_session('u2net', providers=['CPUExecutionProvider']); "
            "print('U2Net CPU model ready')"
        )
        run([str(python), "-c", model_probe], "下载并初始化 U2Net CPU 模型")

    print("\n==> 安装技能文件")
    copy_payload(target)
    runtime_config = {
        "python": str(python),
        "runtime_dir": str(runtime_dir),
        "skill_dir": str(target),
        "model_prefetched": not args.skip_model,
    }
    (target / "runtime.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    verify_command = [str(python), str(target / "scripts" / "verify_install.py")]
    if not args.skip_model:
        verify_command.append("--check-model")
    run(verify_command, "运行 PSD 图层、透明蒙版和纯黑背景自检")

    print("\n安装完成。请重新打开 Codex 或新建任务，然后说：")
    print("  使用 PS_cutout image 技能处理这张图片")
    print(f"技能路径：{target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\n安装失败：步骤返回状态码 {exc.returncode}。", file=sys.stderr)
        print("请保留上方错误信息，并查看 README.md 的故障排查部分。", file=sys.stderr)
        raise SystemExit(exc.returncode)

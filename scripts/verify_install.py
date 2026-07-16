#!/usr/bin/env python3
"""Verify imports, layered PSD writing, alpha retention, and black output."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rice_black_bg_finalize import (  # noqa: E402
    compose_on_background,
    inspect_layered_psd,
    save_layered_psd,
)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 PS_cutout image 运行环境")
    parser.add_argument("--check-model", action="store_true", help="同时初始化 U2Net CPU 模型")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "Pillow",
                "scipy",
                "opencv-python-headless",
                "psd-tools",
                "rembg",
                "onnxruntime",
            )
        },
        "checks": {},
    }
    checks = report["checks"]
    assert isinstance(checks, dict)

    original = Image.new("RGB", (32, 32), (48, 48, 48))
    rgba_array = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba_array[6:26, 8:24, :3] = (42, 170, 72)
    rgba_array[6:26, 8:24, 3] = 255
    rgba = Image.fromarray(rgba_array).convert("RGBA")

    with tempfile.TemporaryDirectory(prefix="ps-cutout-verify-") as temp_dir:
        psd_path = Path(temp_dir) / "synthetic_cutout.psd"
        save_layered_psd(original, rgba, psd_path)
        psd_report = inspect_layered_psd(psd_path)
        checks["psd_exists"] = psd_path.exists() and psd_path.stat().st_size > 0
        checks["required_layers_present"] = bool(psd_report.get("required_layers_present"))
        checks["layer_visibility_ok"] = bool(psd_report.get("visible_delivery_layers_ok"))
        checks["cutout_alpha_ok"] = bool(psd_report.get("cutout_layer_alpha_ok"))
        checks["black_background_layer_rgb_000"] = bool(
            psd_report.get("black_background_rgb_000_ok")
        )

        black = compose_on_background(rgba, (0, 0, 0))
        black_array = np.asarray(black)
        outside = rgba_array[:, :, 3] == 0
        checks["png_background_rgb_000"] = bool((black_array[outside] == 0).all())

    if args.check_model:
        from rembg import new_session

        session = new_session("u2net", providers=["CPUExecutionProvider"])
        inner = getattr(session, "inner_session", None)
        providers = inner.get_providers() if inner is not None else []
        report["model_providers"] = providers
        checks["u2net_cpu_model_ready"] = "CPUExecutionProvider" in providers

    report["status"] = "pass" if all(bool(value) for value in checks.values()) else "fail"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

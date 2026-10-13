# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""ultralytics 单版本训练入口: python _run_ultra.py <name> <model> <batch>
注意: Windows 多进程 spawn 必须有 __main__ 保护，否则 dataloader worker 会递归复制自己
"""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）


def main():
    name, model_file, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
    m = YOLO(model_file)
    m.train(
        data=str(ROOT / "dataset/inspection/data.yaml"),
        epochs=100, imgsz=640, batch=batch, workers=2, device=0,
        project=str(ROOT / "runs_train"), name=name, exist_ok=True,
        seed=0, patience=30, plots=True,
    )
    print(f"DONE {name}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""YOLOv13 训练入口（在 yolov13 仓库目录下、用 envs/yolo13 环境执行）"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）


def main():
    m = YOLO("yolov13n.pt")
    m.train(
        data=str(ROOT / "dataset/inspection/data.yaml"),
        epochs=100, imgsz=640, batch=8, workers=0, device=0,
        project=str(ROOT / "runs_train"), name="yoloV13", exist_ok=True,
        seed=0, patience=30, plots=True,
    )
    print("DONE yoloV13")


if __name__ == "__main__":
    main()

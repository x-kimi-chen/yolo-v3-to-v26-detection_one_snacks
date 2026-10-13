# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
12 版本推理对比：对同一张验证集图片跑各版本 best 权重，输出单格渲染图
用法:
  主环境:  python scripts/showcase_infer.py main      (跑 v3/v5/v6/v8/v9/v10/v11/v12/v26)
  v13环境: python scripts/showcase_infer.py v13       (在 yolov13 仓库目录下用 envs/yolo13 跑)
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
IMG = ROOT / "dataset/inspection/images/val/IMG20261012143655.jpg"
CELLS = ROOT / "output/_cells"
CELLS.mkdir(parents=True, exist_ok=True)

MAIN = [
    ("yoloV3",  ROOT / "runs_train/yoloV3/weights/best.pt",  0.982),
    ("yoloV5",  ROOT / "runs_train/yoloV5/weights/best.pt",  0.979),
    ("yoloV6",  ROOT / "runs_train/yoloV6/weights/best.pt",  0.966),
    ("yoloV8",  ROOT / "runs_train/yoloV8/weights/best.pt",  0.980),
    ("yoloV9",  ROOT / "runs_train/yoloV9/weights/best.pt",  0.977),
    ("yoloV10", ROOT / "runs_train/yoloV10/weights/best.pt", 0.978),
    ("yoloV11", ROOT / "runs_train/yoloV11/weights/best.pt", 0.980),
    ("yoloV12", ROOT / "runs_train/yoloV12/weights/best.pt", 0.983),
    ("yoloV26", ROOT / "runs_train/yoloV26/weights/best.pt", 0.983),
]
V13 = [("yoloV13", ROOT / "runs_train/yoloV13/weights/best.pt", 0.979)]

COLORS = {0: (230, 130, 0), 1: (180, 105, 255), 2: (0, 0, 230), 3: (0, 180, 0), 4: (40, 170, 240)}
CLASSES = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]


def render(name, result, mAP, W=560):
    from PIL import Image
    img0 = cv2.imread(str(IMG))
    H0, W0 = img0.shape[:2]
    s = W / max(H0, W0)
    vis = cv2.resize(img0, (int(W0 * s), int(H0 * s)))
    for b in result.boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        cls = int(b.cls[0]); conf = float(b.conf[0])
        x1, y1, x2, y2 = (xyxy * s).astype(int)
        col = COLORS.get(cls, (255, 0, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = f"{CLASSES[cls]} {conf:.2f}" if conf < 0.99 else CLASSES[cls]
        cv2.putText(vis, label, (x1 + 3, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    # 顶部信息条
    bar = np.full((46, vis.shape[1], 3), 30, np.uint8)
    vis = np.vstack([bar, vis])
    cv2.putText(vis, f"{name}  mAP@0.5={mAP:.3f}", (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out = CELLS / f"{name}.jpg"
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{name}: {len(result.boxes)} 框 -> {out.name}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    from ultralytics import YOLO
    models = MAIN if mode == "main" else V13
    for name, w, mAP in models:
        m = YOLO(str(w))
        r = m.predict(str(IMG), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        render(name, r, mAP)
    print(f"[{mode}] 完成")


if __name__ == "__main__":
    main()

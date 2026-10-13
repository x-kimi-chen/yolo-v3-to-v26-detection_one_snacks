# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""合成 12 版本推理对比宫格图"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
CELLS = ROOT / "output/_cells"
OUT = ROOT / "output/inference_compare_12.jpg"

WIDTH = 560
BAR_H = 48

ORDER = [
    ("yoloV3", 0.982, None), ("yoloV4", 0.944, CELLS / "v4_raw.jpg"),
    ("yoloV5", 0.979, None), ("yoloV6", 0.966, None),
    ("yoloV7", 0.978, CELLS / "v7tmp/IMG20261012143655.jpg"), ("yoloV8", 0.980, None),
    ("yoloV9", 0.977, None), ("yoloV10", 0.978, None),
    ("yoloV11", 0.980, None), ("yoloV12", 0.983, None),
    ("yoloV13", 0.979, None), ("yoloV26", 0.983, None),
]


def load_cell(name, mAP, raw_path):
    path = raw_path if raw_path else CELLS / f"{name}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        print("缺少:", path)
        return None
    # 统一到 WIDTH 宽
    s = WIDTH / img.shape[1]
    img = cv2.resize(img, (WIDTH, int(img.shape[0] * s)))
    if raw_path:
        # 原始渲染图没有顶栏，补一个
        bar = np.full((BAR_H, WIDTH, 3), 30, np.uint8)
        cv2.putText(bar, f"{name}  mAP@0.5={mAP:.3f}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        img = np.vstack([bar, img])
    return img


cells = []
for name, mAP, raw in ORDER:
    c = load_cell(name, mAP, raw)
    if c is not None:
        cells.append(c)

max_h = max(c.shape[0] for c in cells)
cols, gap, pad = 4, 14, 24
title_h = 78
rows = (len(cells) + cols - 1) // cols
canvas_w = cols * WIDTH + (cols + 1) * gap + pad * 2
canvas_h = title_h + rows * max_h + (rows + 1) * gap + pad
canvas = np.full((canvas_h, canvas_w, 3), 245, np.uint8)

cv2.putText(canvas, "YOLOv3 -> YOLO26  Full-Version Inference Comparison (same val image)",
            (pad, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (20, 20, 20), 2)

for i, c in enumerate(cells):
    r, cc = divmod(i, cols)
    x0 = pad + gap + cc * (WIDTH + gap)
    y0 = title_h + gap + r * (max_h + gap)
    canvas[y0:y0 + c.shape[0], x0:x0 + WIDTH] = c

cv2.imwrite(str(OUT), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("合成完成:", OUT, canvas.shape)

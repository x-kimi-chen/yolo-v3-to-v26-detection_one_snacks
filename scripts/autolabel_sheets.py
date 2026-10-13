# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""重新生成清晰版聚类代表图（每类最多36个代表性样本）"""
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
WORK = ROOT / "autolabel"
SRC = ROOT / "my_images"

proposals = {}
for line in (WORK / "proposals.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        proposals[d["name"]] = d
clusters = json.loads((WORK / "clusters.json").read_text(encoding="utf-8"))

# 读取每个框的缩略图
cache = {}
def thumb_of(name, bidx):
    key = (name, bidx)
    if key not in cache:
        rec = proposals[name]
        b = rec["boxes"][bidx]["box"]
        x1, y1, x2, y2 = [int(v) for v in b]
        img = cv2.imread(str(SRC / name))
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            cache[key] = np.zeros((170, 190, 3), np.uint8)
        else:
            cache[key] = cv2.resize(crop, (190, 170))
    return cache[key]

def sheet(items, out_path, title):
    items = items[:36]
    cols = 6
    cw, chh = 200, 210
    rows = max(1, (len(items) + cols - 1) // cols)
    canvas = np.full((rows * chh + 50, cols * cw, 3), 255, np.uint8)
    cv2.putText(canvas, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    for i, it in enumerate(items):
        t = thumb_of(it["name"], it["bidx"])
        r, c = divmod(i, cols)
        y0, x0 = 50 + r * chh, c * cw
        canvas[y0:y0 + 170, x0:x0 + 190] = t
        cv2.putText(canvas, f"{it['name'][:12]}#{it['bidx']}", (x0 + 2, y0 + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), canvas)
    print("生成:", out_path)

# 各聚类
for j in range(6):
    its = [it for it in clusters if it["cluster"] == j]
    its.sort(key=lambda t: t["dist"])  # 离中心最近的最具代表性
    print(f"Cluster {j}: {len(its)} 个")
    sheet(its, WORK / f"sheet_{j}.jpg", f"Cluster {j}   n={len(its)}")

# 离群
ds = sorted([t["dist"] for t in clusters])
thr = ds[int(len(ds) * 0.92)]
outl = [t for t in clusters if t["dist"] > thr]
outl.sort(key=lambda t: -t["dist"])
sheet(outl, WORK / "sheet_outliers.jpg", f"Outliers  n={len(outl)}")
print("完成")

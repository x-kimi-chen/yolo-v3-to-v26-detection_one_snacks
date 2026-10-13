# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""从 proposals.jsonl 缓存渲染指定图片的框（质检用）
用法: python autolabel_render_cache.py 图片名1 图片名2 ...
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
WORK = ROOT / "autolabel"
SRC = ROOT / "my_images"
OUT = WORK / "test"
OUT.mkdir(exist_ok=True)

cache = {}
for line in (WORK / "proposals.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        cache[d["name"]] = d  # 同名后者覆盖（缓存重写后无重复）

for name in sys.argv[1:]:
    rec = cache.get(name)
    if rec is None:
        print(f"{name}: 不在缓存中")
        continue
    img = cv2.imread(str(SRC / name))
    W, H = rec["W"], rec["H"]
    s = 900 / max(W, H)
    vis = cv2.resize(img, (int(W * s), int(H * s)))
    for k, b in enumerate(rec["boxes"]):
        x1, y1, x2, y2 = (np.array(b["box"]) * s).astype(int)
        col = (0, 0, 255) if b.get("src") != "surgery" else (255, 0, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        cv2.putText(vis, str(k), (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    cv2.imwrite(str(OUT / f"qc_{name}"), vis)
    print(f"{name}: {len(rec['boxes'])}框 (块{rec['n_blobs']}/救{rec['n_rescued']}) -> qc_{name}")

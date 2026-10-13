# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
最终定级 + 导出 YOLO 预标注
基于 refine.json 的像素特征，用经验证的规则定类，然后:
  - 每图每类去重（保留得分最高，最多2个）
  - 写出 autolabel/prelabels/*.txt 和 autolabel/previews/*.jpg
"""
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
WORK = ROOT / "autolabel"
SRC = ROOT / "my_images"
PRELABELS = WORK / "prelabels"
PREVIEWS = WORK / "previews"
CLASSES = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
CIDX = {c: i for i, c in enumerate(CLASSES)}
COLORS = {0: (230, 130, 0), 1: (180, 105, 255), 2: (0, 0, 230), 3: (0, 180, 0), 4: (40, 170, 240)}

props = {}
for line in (WORK / "proposals.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        props[d["name"]] = d
recs = json.loads((WORK / "refine.json").read_text(encoding="utf-8"))


def classify(r):
    c, f = r["cluster"], r
    red, green, gold, blue, pink, lowS, edge = f["red"], f["green"], f["gold"], f["blue"], f["pink"], f["lowS"], f["edge"]
    # 饼干：全局规则（含垃圾簇）；边缘密度负责把木桌纹理过滤掉
    if gold >= 0.33 and edge >= 0.05 and red < 0.30:
        return "cracker", gold
    # 绿袋：全局规则
    if 0.18 <= green <= 0.40 and red >= 0.20 and edge >= 0.055:
        return "gluten_pepper", edge + green
    # 纸箱印刷簇 只允许上面的两类
    if c in (3, 4):
        return None, 0
    # 反光板簇：只有蓝色部分可用
    if c == 1:
        return ("puff_vanilla", blue) if blue >= 0.30 else (None, 0)
    # 蓝袋：聚类2是最可信的蓝色集合（浅蓝饱和度低，宽容处理）
    if c == 2 and blue >= 0.06 and lowS < 0.95:
        return "puff_vanilla", blue + 0.2
    if blue >= 0.22 and lowS < 0.85:
        return "puff_vanilla", blue
    # 红袋 vs 粉袋：用 red-pink 差值判别（已校准：红袋差值中位0.62，粉袋-0.27）
    if red >= 0.35 and red - pink >= 0.20:
        return "gluten_spicy", red
    if pink >= 0.12 or pink - red >= 0.10:
        return "puff_strawberry", pink
    if green >= 0.15 and red >= 0.25 and edge >= 0.055:
        return "gluten_pepper", edge + green
    return None, 0


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


# 定级
for r in recs:
    cls, score = classify(r)
    r["final"] = cls
    r["score"] = round(score, 4)

# 每图每类 去重+限量
from collections import defaultdict
groups = defaultdict(list)
for r in recs:
    if r["final"]:
        groups[(r["name"], r["final"])].append(r)

final_by_img = defaultdict(list)
dups = 0
for (name, cls), rs in groups.items():
    rs.sort(key=lambda r: -r["score"])
    kept = []
    for r in rs:
        b = props[name]["boxes"][r["bidx"]]["box"]
        too_close = any(iou(b, k) > 0.35 for k, _ in kept)
        if too_close:
            dups += 1
            continue
        if len(kept) >= 2:
            dups += 1
            continue
        kept.append((b, r))
    for b, r in kept:
        final_by_img[name].append((r["final"], b, r["score"]))

# 导出
PRELABELS.mkdir(exist_ok=True)
PREVIEWS.mkdir(exist_ok=True)

# ---------- 补救：对缺失类别用放宽阈值找最佳候选（每类最多补1个） ----------
RELAX = {
    "puff_vanilla":    (lambda f: f["blue"] >= 0.10 and f["lowS"] < 0.75 and f["edge"] >= 0.012, "blue"),
    "puff_strawberry": (lambda f: f["pink"] >= 0.09 and f["edge"] >= 0.010, "pink"),
    "gluten_spicy":    (lambda f: f["red"] >= 0.30 and (f["red"] - f["pink"]) >= 0.15, "red"),
    "gluten_pepper":   (lambda f: 0.12 <= f["green"] <= 0.42 and f["red"] >= 0.16 and f["edge"] >= 0.045, "green"),
    "cracker":         (lambda f: f["gold"] >= 0.28 and f["edge"] >= 0.040 and f["red"] < 0.32, "gold"),
}
have_cls = defaultdict(set)
for name, items in final_by_img.items():
    for cls, b, sc in items:
        have_cls[name].add(cls)

from collections import defaultdict as _dd
by_img_recs = _dd(list)
for r in recs:
    by_img_recs[r["name"]].append(r)

repair_added = defaultdict(int)
for name, rs in by_img_recs.items():
    accepted = [b for cls, b, sc in final_by_img.get(name, [])]
    for cls, (fn, key) in RELAX.items():
        if cls in have_cls[name]:
            continue
        cands = []
        for r in rs:
            b = props[name]["boxes"][r["bidx"]]["box"]
            if r.get("final") or not (0.025 <= r["area"] <= 0.25):
                continue
            if b[2] - b[0] < 40 or b[3] - b[1] < 40:
                continue
            if fn(r):
                cands.append((r[key], b))
        if not cands:
            continue
        cands.sort(key=lambda t: -t[0])
        pool = accepted[:]
        for sc, b in cands[:3]:
            if all(iou(b, a) < 0.5 for a in pool):
                final_by_img[name].append((cls, b, round(sc * 0.5, 4)))
                pool.append(b)
                repair_added[cls] += 1
                break
print("补救新增:", dict(repair_added), "共", sum(repair_added.values()))
# ---------- 补救结束 ----------

img_cache = {}
cls_count = defaultdict(int)
per_img_count = []
no_box = []
for name in sorted(props):
    items = final_by_img.get(name, [])
    W, H = props[name]["W"], props[name]["H"]
    lines = []
    for cls, b, sc in items:
        x1, y1, x2, y2 = b
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 30 or y2 - y1 < 30:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{CIDX[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        cls_count[cls] += 1
    per_img_count.append(len(lines))
    (PRELABELS / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not lines:
        no_box.append(name)
    # 预览
    if name not in img_cache:
        img_cache[name] = cv2.imread(str(SRC / name))
    img = img_cache[name]
    s = 900 / max(W, H)
    vis = cv2.resize(img, (int(W * s), int(H * s)))
    for cls, b, sc in items:
        x1, y1, x2, y2 = (np.array(b) * s).astype(int)
        col = COLORS[CIDX[cls]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, cls, (x1 + 3, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(str(PREVIEWS / name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

print(f"分类去重丢弃: {dups}")
(WORK / "final.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
print(f"总框: {sum(cls_count.values())}, 平均 {np.mean(per_img_count):.1f}/图")
for c in CLASSES:
    print(f"  {c}: {cls_count.get(c, 0)}")
print(f"无框图片: {len(no_box)}")
import collections
hist = collections.Counter(per_img_count)
print("每图框数分布:", dict(sorted(hist.items())))

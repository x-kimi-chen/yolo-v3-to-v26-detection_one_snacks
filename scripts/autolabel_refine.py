# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
精炼分类：像素级主色规则 + 同类去重，生成待核查清单图
输出:
  autolabel/refine.json     每个框的最终类别 (name, bidx, cls, fracs...)
  autolabel/verify_cracker.jpg / verify_green.jpg / verify_c1blue.jpg  待人工核查
"""
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
WORK = ROOT / "autolabel"
SRC = ROOT / "my_images"

CLASSES = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

props = {}
for line in (WORK / "proposals.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        props[d["name"]] = d
clusters = json.loads((WORK / "clusters.json").read_text(encoding="utf-8"))
clu = {(c["name"], c["bidx"]): c["cluster"] for c in clusters}

imgs = {}
def get_img(name):
    if name not in imgs:
        imgs[name] = cv2.imread(str(SRC / name))
    return imgs[name]

def analyze(name, bidx):
    b = props[name]["boxes"][bidx]["box"]
    img = get_img(name)
    x1, y1, x2, y2 = [int(v) for v in b]
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    n = H.size

    def frac(lo, hi, smin, vmin=0):
        m = ((H >= lo) & (H < hi)) if lo < hi else ((H >= lo) | (H < hi))
        return float((m & (S > smin) & (V > vmin)).sum()) / n

    f = {
        "red": frac(172, 8, 90),
        "pink": frac(140, 172, 30),
        "blue": frac(85, 135, 40),
        "green": frac(40, 85, 70),
        "gold": frac(8, 40, 60, 110),
        "lowS": float((S < 25).sum()) / n,
    }
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge = float((cv2.Canny(g, 60, 150) > 0).mean())
    f["edge"] = edge
    f["area"] = (x2 - x1) * (y2 - y1) / (props[name]["W"] * props[name]["H"])
    return f


def prelim_class(c, f):
    if c in (3, 4):
        return None
    if c == 1:
        return "puff_vanilla" if f["blue"] > 0.35 and f["edge"] > 0.02 else None
    if c == 2:
        return "puff_vanilla" if f["lowS"] < 0.85 else None
    if c == 5:
        return "puff_strawberry"
    if c == 0:
        if f["green"] > 0.28 and f["edge"] > 0.025:
            return "gluten_pepper"
        if f["red"] > 0.30:
            return "gluten_spicy"
        if f["gold"] > 0.32 and f["edge"] > 0.02:
            return "cracker"
        return None
    # 离群
    if f["blue"] > 0.40 and f["edge"] > 0.03:
        return "puff_vanilla"
    if f["gold"] > 0.35 and f["edge"] > 0.025 and f["red"] < 0.2:
        return "cracker"
    if f["green"] > 0.30 and f["edge"] > 0.03:
        return "gluten_pepper"
    return None


records = []
for (name, bidx) in clu:
    f = analyze(name, bidx)
    if f is None:
        continue
    c = clu[(name, bidx)]
    cls = prelim_class(c, f)
    rec = {"name": name, "bidx": bidx, "cluster": c, "cls": cls, **{k: round(v, 4) for k, v in f.items()}}
    records.append(rec)

# 全局垃圾过滤
for r in records:
    if r["lowS"] > 0.75 and r["edge"] < 0.02:
        r["cls"] = None
    if r["area"] < 0.018:
        r["cls"] = None

# 同类去重（每图内 IoU>0.2 合并，保留主色纯度最高者）
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)

by_img_cls = {}
for r in records:
    if r["cls"]:
        by_img_cls.setdefault((r["name"], r["cls"]), []).append(r)

dropped = 0
for (name, cls), rs in by_img_cls.items():
    if len(rs) < 2:
        continue
    boxes = [props[name]["boxes"][r["bidx"]]["box"] for r in rs]
    # 找重叠组
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if rs[i]["cls"] and rs[j]["cls"] and iou(boxes[i], boxes[j]) > 0.2:
                # 保留纯度更高的
                key = {"puff_vanilla": "blue", "puff_strawberry": "pink", "gluten_spicy": "red",
                       "gluten_pepper": "green", "cracker": "gold"}[cls]
                pi = rs[i].get(key, 0)
                pj = rs[j].get(key, 0)
                loser = rs[j] if pi >= pj else rs[i]
                if loser["cls"]:
                    loser["cls"] = None
                    dropped += 1

n_cls = {}
for r in records:
    if r["cls"]:
        n_cls[r["cls"]] = n_cls.get(r["cls"], 0) + 1
print("去重丢弃:", dropped)
print("类别统计:", n_cls)

(WORK / "refine.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
print("已保存 refine.json")

# ---- 核查清单图 ----
def sheet(rs, path, title, key_show):
    cols = 6
    cw, chh = 200, 210
    rs = rs[:36]
    rows = max(1, (len(rs) + cols - 1) // cols)
    canvas = np.full((rows * chh + 50, cols * cw, 3), 255, np.uint8)
    cv2.putText(canvas, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, r in enumerate(rs):
        b = props[r["name"]]["boxes"][r["bidx"]]["box"]
        img = get_img(r["name"])
        x1, y1, x2, y2 = [int(v) for v in b]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        t = cv2.resize(crop, (190, 170))
        rr, cc = divmod(i, cols)
        y0, x0 = 50 + rr * chh, cc * cw
        canvas[y0:y0 + 170, x0:x0 + 190] = t
        cv2.putText(canvas, f"{r['name'][-14:]}#{r['bidx']} {key_show(r)}", (x0 + 2, y0 + 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)
    print("生成:", path, f"({len(rs)}个)")

# 1) 饼干候选（全库金色主导）
cracker_cands = sorted([r for r in records if r["gold"] > 0.30 and r["edge"] > 0.018 and r["lowS"] < 0.5],
                       key=lambda r: -r["gold"])
sheet(cracker_cands, WORK / "verify_cracker.jpg", f"Cracker candidates n={len(cracker_cands)}",
      lambda r: f"g{r['gold']:.2f} e{r['edge']:.2f}")

# 2) 绿袋候选
green_cands = sorted([r for r in records if r["green"] > 0.25 and r["edge"] > 0.02],
                     key=lambda r: -r["green"])
sheet(green_cands, WORK / "verify_green.jpg", f"Green candidates n={len(green_cands)}",
      lambda r: f"gr{r['green']:.2f} e{r['edge']:.2f}")

# 3) 簇1里的蓝袋
c1blue = [r for r in records if r["cluster"] == 1 and r["blue"] > 0.30]
sheet(c1blue, WORK / "verify_c1blue.jpg", f"Cluster1 blue n={len(c1blue)}",
      lambda r: f"b{r['blue']:.2f}")

# 4) 绿袋/饼干在垃圾簇(3/4)里的候选
c34 = sorted([r for r in records if r["cluster"] in (3, 4) and (r["green"] > 0.30 or (r["gold"] > 0.35 and r["edge"] > 0.03))],
             key=lambda r: -(r["green"] + r["gold"]))
sheet(c34, WORK / "verify_c34.jpg", f"Cluster3/4 product-like n={len(c34)}",
      lambda r: f"gr{r['green']:.2f} g{r['gold']:.2f} e{r['edge']:.2f}")
print("完成")

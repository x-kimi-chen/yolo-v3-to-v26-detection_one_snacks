# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
自动预标注主管线（阶段式，可断点续跑）
  阶段1 proposals: FastSAM 双参数提案 -> 过滤 -> 可疑大块放大重分割 -> 缓存到 proposals.jsonl
  阶段2 clusters : 提取每个框的颜色特征 -> KMeans 聚类 -> 生成聚类代表图（供人工确认类别）
用法:
  python autolabel_run.py proposals
  python autolabel_run.py clusters [--k 6]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from ultralytics import FastSAM

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
SRC = ROOT / "my_images"
WORK = ROOT / "autolabel"
WORK.mkdir(exist_ok=True)
CACHE = WORK / "proposals.jsonl"
TMP_CROP = WORK / "_tmp_crop.jpg"

model = FastSAM(str(ROOT / "FastSAM-s.pt"))


# ---------------- 阶段1: 提案 ----------------
def run_fastsam(img_path):
    r = model(str(img_path), device=0, imgsz=1024, conf=0.35, iou=0.9, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    out = []
    for i, b in enumerate(boxes):
        sol = None
        if masks is not None:
            m = masks[i].astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) > 0:
                ma = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
                sol = float(m.sum()) / max(ma, 1)
        out.append({"box": [float(v) for v in b], "sol": sol})
    return out


def dedupe(cands):
    """按面积从大到小，去掉与已保留框高度重叠/包含的"""
    def im_ratio_iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / min(aa, ab), inter / (aa + ab - inter)

    cands = sorted(cands, key=lambda c: -((c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])))
    kept = []
    for c in cands:
        drop = False
        for k in kept:
            imr, iou = im_ratio_iou(c["box"], k["box"])
            if imr > 0.75 or iou > 0.5:
                drop = True
                break
        if not drop:
            kept.append(c)
    return kept


def filter_props(cands, W, H):
    ia = W * H
    kept, blobs = [], []
    for c in cands:
        x1, y1, x2, y2 = c["box"]
        bw, bh = x2 - x1, y2 - y1
        if bw < 80 or bh < 80:
            continue
        ar = bw * bh / ia
        if ar < 0.02:
            continue
        if ar > 0.22:
            if ar < 0.55:
                blobs.append(c)
            continue
        if c["sol"] is not None and c["sol"] < 0.55:
            continue
        kept.append(c)
    return dedupe(kept), blobs


def blob_surgery(img_path, blob, W, H):
    """对可疑大块裁剪放大后重新分割，尝试拆出里面的物品（严格限制：子框必须明显小于大块）"""
    x1, y1, x2, y2 = blob["box"]
    bw, bh = x2 - x1, y2 - y1
    blob_area = bw * bh
    px, py = 0.08 * bw, 0.08 * bh
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
    if cx2 - cx1 < 100 or cy2 - cy1 < 100:
        return []
    img = cv2.imread(str(img_path))
    crop = img[cy1:cy2, cx1:cx2]
    cv2.imwrite(str(TMP_CROP), crop)
    sub = run_fastsam(TMP_CROP)
    cw, ch = cx2 - cx1, cy2 - cy1
    out = []
    for s in sub:
        sx1, sy1, sx2, sy2 = s["box"]
        ar_crop = (sx2 - sx1) * (sy2 - sy1) / (cw * ch)
        ar_img = (sx2 - sx1) * (sy2 - sy1) / (W * H)
        # 子框: 块内占比 8%~60%, 全图占比 ≤18%, 且面积不超过大块的 65%
        if not (0.08 < ar_crop < 0.60):
            continue
        if ar_img > 0.18:
            continue
        box_img = [sx1 + cx1, sy1 + cy1, sx2 + cx1, sy2 + cy1]
        if (sx2 - sx1) * (sy2 - sy1) > 0.65 * blob_area:
            continue
        out.append({"box": box_img, "sol": s["sol"], "src": "surgery"})
    out = dedupe(out)[:8]  # 每块最多保留8个子框
    return out


def stage_proposals():
    files = sorted(f for f in SRC.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["name"]] = d
    todo = [f for f in files if f.name not in cache]
    print(f"总图片 {len(files)}, 已缓存 {len(cache)}, 待处理 {len(todo)}", flush=True)

    t0 = time.time()
    fp = open(CACHE, "a", encoding="utf-8")
    for n, f in enumerate(todo, 1):
        img = Image.open(f)
        W, H = img.size
        c1 = run_fastsam(f)
        kept, blobs = filter_props(c1, W, H)
        # 大块救援
        rescued = []
        for b in blobs:
            subs = blob_surgery(f, b, W, H)
            rescued.extend(subs)

        # 救援框只用于补充基础框未覆盖的区域
        def overlaps(a, b):
            ax1, ay1 = max(a[0], b[0]), max(a[1], b[1])
            ax2, ay2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            aa = (a[2] - a[0]) * (a[3] - a[1])
            ab = (b[2] - b[0]) * (b[3] - b[1])
            return inter / min(aa, ab) > 0.6 or inter / (aa + ab - inter) > 0.45

        extra = [r for r in rescued if not any(overlaps(r["box"], k["box"]) for k in kept)]
        final = dedupe(kept + extra)
        rec = {"name": f.name, "W": W, "H": H,
               "boxes": final, "n_blobs": len(blobs), "n_rescued": len(rescued)}
        cache[f.name] = rec
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s  最近一张: {f.name} 框{len(final)}(块{len(blobs)}/救{len(rescued)})", flush=True)
    fp.close()

    with open(CACHE, "w", encoding="utf-8") as fp:
        for name in sorted(cache):
            fp.write(json.dumps(cache[name], ensure_ascii=False) + "\n")

    tot = sum(len(c["boxes"]) for c in cache.values())
    nb = sum(c["n_blobs"] for c in cache.values())
    nr = sum(c["n_rescued"] for c in cache.values())
    print(f"\n完成: {len(cache)} 张, 总框数 {tot}, 平均 {tot/len(cache):.1f} 框/张")
    print(f"可疑大块 {nb} 个, 成功救援 {nr} 个子框")
    low = [c["name"] for c in cache.values() if len(c["boxes"]) < 3]
    high = [c["name"] for c in cache.values() if len(c["boxes"]) > 8]
    if low:
        print(f"[关注] 框数<3 的图 {len(low)} 张: {low[:10]}")
    if high:
        print(f"[关注] 框数>8 的图 {len(high)} 张: {high[:10]}")


# ---------------- 阶段2: 颜色特征 + 聚类 ----------------
def crop_feature(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(0.02 * bw), int(0.02 * bh)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(img.shape[1], x2 + px), min(img.shape[0], y2 + py)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten(); h /= h.sum() + 1e-9
    s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten(); s /= s.sum() + 1e-9
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten(); v /= v.sum() + 1e-9
    feat = np.concatenate([h * 1.0, s * 0.5, v * 0.5]).astype(np.float32)
    # 缩略图给聚类代表图用（固定尺寸）
    thumb = cv2.resize(crop, (140, 130))
    return feat, thumb


def kmeans(X, k, iters=60, seed=0):
    rng = np.random.RandomState(seed)
    centers = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        p = d / (d.sum() + 1e-12)
        centers.append(X[rng.choice(len(X), p=p)])
    centers = np.array(centers)
    lab = None
    for _ in range(iters):
        d = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], 1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                centers[j] = X[lab == j].mean(0)
    d = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], 1)
    return lab, d.min(1), centers


def stage_clusters(k):
    cache = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for rec in cache:
        img = cv2.imread(str(SRC / rec["name"]))
        for bi, b in enumerate(rec["boxes"]):
            feat, thumb = crop_feature(img, b["box"])
            if feat is None:
                continue
            items.append({"name": rec["name"], "bidx": bi, "feat": feat, "thumb": thumb, "box": b["box"]})
    print(f"共 {len(items)} 个框特征")
    X = np.stack([it["feat"] for it in items])
    lab, dist, centers = kmeans(X, k)
    for it, l, d in zip(items, lab, dist):
        it["cluster"] = int(l)
        it["dist"] = float(d)
    # 离群值: 距离最大的 8%
    thr = np.quantile(dist, 0.92)
    outliers = [it for it in items if it["dist"] > thr]
    print(f"离群候选: {len(outliers)}")

    # 每个聚类画代表图
    def sheet(its, out_path, title):
        cols = 6
        cell_w, cell_h = 150, 180
        rows = (len(its) + cols - 1) // cols
        canvas = np.full((rows * cell_h + 40, cols * cell_w, 3), 255, np.uint8)
        cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        for i, it in enumerate(its[:cols * rows]):
            t = it["thumb"]
            r, c = divmod(i, cols)
            y0, x0 = 40 + r * cell_h, c * cell_w
            canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
            cv2.putText(canvas, f"{it['name'][:10]}#{it['bidx']}", (x0 + 2, y0 + t.shape[0] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.imwrite(str(out_path), canvas)
        print("生成:", out_path)

    for j in range(k):
        its = [it for it in items if it["cluster"] == j]
        its.sort(key=lambda t: t["dist"])
        sheet(its, WORK / f"cluster_{j}.jpg", f"Cluster {j}  n={len(its)}")
    out = [{"name": it["name"], "bidx": it["bidx"], "box": it["box"],
            "cluster": it["cluster"], "dist": it["dist"]} for it in items]
    (WORK / "clusters.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已保存: autolabel/clusters.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["proposals", "clusters"])
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    if args.stage == "proposals":
        stage_proposals()
    else:
        stage_clusters(args.k)

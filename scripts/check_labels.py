# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
标注质量抽查：把 YOLO 标签画回图片上，随机抽若干张存到 _check 目录，肉眼快速检查标注质量
用法: python scripts/check_labels.py [--n 12]
"""
import argparse, random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
DATASET = ROOT / "dataset/inspection"


def draw(img_path: Path, lbl_path: Path, out_path: Path, names):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            c, x, y, bw, bh = line.split()[:5]
            c = int(float(c)); x, y, bw, bh = map(float, (x, y, bw, bh))
            x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
            x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, w // 500))
            label = names[c] if c < len(names) else str(c)
            cv2.putText(img, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1600), (0, 0, 255), max(1, w // 800))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="抽查图片数量")
    args = ap.parse_args()

    import yaml
    data = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    names = [data["names"][i] for i in sorted(data["names"])]

    imgs = []
    for split in ("train", "val"):
        for p in (DATASET / "images" / split).glob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                imgs.append((split, p))

    random.seed(0)
    sample = random.sample(imgs, min(args.n, len(imgs)))
    out_dir = DATASET / "_check"
    print(f"抽查 {len(sample)} 张 -> {out_dir}")
    for split, p in sample:
        lbl = DATASET / "labels" / split / f"{p.stem}.txt"
        out = out_dir / f"{split}_{p.name}"
        ok = draw(p, lbl, out, names)
        n_box = len([l for l in lbl.read_text().splitlines() if l.strip()]) if lbl.exists() else 0
        print(f"  [{split}] {p.name}  框数={n_box}  {'已生成' if ok else '读取失败'}")
    print("\n请打开 _check 目录肉眼核对：框是否贴合目标、类别是否正确、有无漏标")


if __name__ == "__main__":
    main()

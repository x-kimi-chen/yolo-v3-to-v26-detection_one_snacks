# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
数据集准备脚本：把标注好的图片+标签整理成 YOLO 训练目录结构
- 按"连拍组"划分 train/val（避免连拍近似图同时进入训练和验证造成数据泄漏）
- 自动把标签整理到 images/labels 分离的标准结构
- 校验标签格式，生成 data.yaml

用法:
  python scripts/prepare_dataset.py                    # 默认 80/20 划分
  python scripts/prepare_dataset.py --val-ratio 0.2    # 自定义验证集比例
"""
import argparse, random, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
SRC_IMAGES = ROOT / "my_images"
CLASSES_FILE = ROOT / "classes.txt"
OUT = ROOT / "dataset/inspection"

# X-AnyLabeling 默认导出到图片目录的同级 labels 文件夹；也兼容标签就在图片旁边的情况
LABEL_SEARCH_DIRS = [ROOT / "labels", SRC_IMAGES]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_key(stem: str) -> str:
    """连拍图分组：IMG20261011133704_01 -> IMG20261011133704"""
    return re.sub(r"_\d+$", "", stem)


def find_label(stem: str):
    for d in LABEL_SEARCH_DIRS:
        p = d / f"{stem}.txt"
        if p.exists():
            return p
    return None


def validate_label(path: Path, n_classes: int) -> list:
    """返回问题列表；空列表=通过"""
    problems = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"第{i}行: 应有5列(cls x y w h)，实际{len(parts)}列")
            continue
        try:
            c, x, y, w, h = int(float(parts[0])), *map(float, parts[1:])
        except ValueError:
            problems.append(f"第{i}行: 数值解析失败")
            continue
        if not (0 <= c < n_classes):
            problems.append(f"第{i}行: 类别id {c} 超出范围 [0,{n_classes-1}]")
        if not all(0 <= v <= 1.001 for v in (x, y, w, h)):
            problems.append(f"第{i}行: 坐标不在[0,1]范围 ({x},{y},{w},{h})")
        if w <= 0 or h <= 0:
            problems.append(f"第{i}行: 宽高必须大于0")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not CLASSES_FILE.exists():
        sys.exit(f"[错误] 找不到类别文件: {CLASSES_FILE}\n请先让 classes.txt 就位（每行一个类名）")
    classes = [l.strip() for l in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_classes = len(classes)
    print(f"类别数: {n_classes} -> {classes}")

    imgs = sorted(p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        sys.exit(f"[错误] {SRC_IMAGES} 里没找到图片")

    # 按组划分
    groups = {}
    for p in imgs:
        groups.setdefault(group_key(p.stem), []).append(p)
    gkeys = sorted(groups)
    random.seed(args.seed)
    random.shuffle(gkeys)
    n_val = max(1, round(len(gkeys) * args.val_ratio))
    val_groups = set(gkeys[:n_val])
    print(f"图片 {len(imgs)} 张, 连拍组 {len(gkeys)} 组 -> 验证集 {n_val} 组 / 训练集 {len(gkeys)-n_val} 组")

    # 建目录（清空旧的，仅限本脚本管理的数据集目录）
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "missing": [], "problems": [], "box_counts": {}}
    for g in sorted(groups):
        split = "val" if g in val_groups else "train"
        for img in groups[g]:
            shutil.copy2(img, OUT / f"images/{split}" / img.name)
            lbl = find_label(img.stem)
            if lbl is None:
                stats["missing"].append(img.name)
                continue
            probs = validate_label(lbl, n_classes)
            if probs:
                stats["problems"].append((lbl.name, probs[:3]))
            shutil.copy2(lbl, OUT / f"labels/{split}" / f"{img.stem}.txt")
            # 统计框数
            for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    c = int(float(line.split()[0]))
                    stats["box_counts"][c] = stats["box_counts"].get(c, 0) + 1
            stats[split] += 1

    # data.yaml
    yaml_lines = [
        f"path: {OUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(classes)]
    (OUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # 报告
    print(f"\n{'='*50}\n划分完成: train {stats['train']} 张 / val {stats['val']} 张")
    print(f"data.yaml -> {OUT / 'data.yaml'}")
    print("\n每类框数统计:")
    for i, name in enumerate(classes):
        print(f"  [{i}] {name}: {stats['box_counts'].get(i, 0)} 个框")
    if stats["missing"]:
        print(f"\n[警告] {len(stats['missing'])} 张图没有标签(未标注或漏标):")
        for n in stats["missing"][:20]:
            print("   ", n)
        if len(stats["missing"]) > 20:
            print(f"    ... 共 {len(stats['missing'])} 张")
    if stats["problems"]:
        print(f"\n[警告] {len(stats['problems'])} 个标签文件有问题:")
        for name, probs in stats["problems"][:20]:
            print(f"   {name}: {probs}")
    if not stats["missing"] and not stats["problems"]:
        print("\n[OK] 全部标签校验通过，无缺失")


if __name__ == "__main__":
    main()

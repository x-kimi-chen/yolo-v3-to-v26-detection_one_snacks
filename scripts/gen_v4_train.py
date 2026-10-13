# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
生成 YOLOv4 (bubbliiiing版) 正式训练配置：
1. 从 dataset/inspection (YOLO格式) 生成 v4 标注文件（绝对坐标 xyxy）
2. 基于原版 train.py 生成 train_real.py（不修改原文件）
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
DATA = ROOT / "dataset/inspection"
REPO = ROOT / "yolo-repos/yolov4-pytorch"
classes_file = ROOT / "classes.txt"
assert classes_file.exists()

def convert(split: str):
    img_dir = DATA / "images" / split
    lbl_dir = DATA / "labels" / split
    out = DATA / f"v4_{split}.txt"
    lines = []
    for img in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        lbl = lbl_dir / f"{img.stem}.txt"
        boxes = []
        if lbl.exists():
            W, H = Image.open(img).size
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = map(float, line.split()[:5])
                x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
                x2, y2 = min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))
                if x2 > x1 and y2 > y1:
                    boxes.append(f"{x1},{y1},{x2},{y2},{int(c)}")
        lines.append(f"{img.as_posix()} " + " ".join(boxes))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}: {len(lines)}张")
    return out

train_txt = convert("train")
val_txt = convert("val")

src = (REPO / "train.py").read_text(encoding="utf-8")
subs = [
    ("    classes_path    = 'model_data/voc_classes.txt'",
     f"    classes_path    = r'{classes_file.as_posix()}'"),
    ("    model_path      = 'model_data/yolo4_weights.pth'",
     "    model_path      = ''"),
    ("    input_shape     = [416, 416]",
     "    input_shape     = [640, 640]"),
    ("    Freeze_Train        = True",
     "    Freeze_Train        = False"),
    ("    Freeze_Epoch        = 50",
     "    Freeze_Epoch        = 1"),
    ("    UnFreeze_Epoch      = 300",
     "    UnFreeze_Epoch      = 100"),
    ("    fp16            = False",
     "    fp16            = True"),
    ("    Unfreeze_batch_size = 4",
     "    Unfreeze_batch_size = 4"),
    ("    save_period         = 10",
     "    save_period         = 10"),
    ("    eval_flag           = True",
     "    eval_flag           = False"),
    ("    optimizer_type      = \"sgd\"",
     "    optimizer_type      = \"adam\""),
    ("    Init_lr             = 1e-2",
     "    Init_lr             = 1e-3"),
    ("    num_workers         = 4",
     "    num_workers         = 2"),
    ("    train_annotation_path   = '2007_train.txt'",
     f"    train_annotation_path   = r'{train_txt.as_posix()}'"),
    ("    val_annotation_path     = '2007_val.txt'",
     f"    val_annotation_path     = r'{val_txt.as_posix()}'"),
    ("    save_dir            = 'logs'",
     f"    save_dir            = r'{(ROOT / 'runs_train/yoloV4').as_posix()}'"),
]
for old, new in subs:
    if old not in src:
        raise SystemExit(f"[错误] 替换目标不存在: {old!r}")
    src = src.replace(old, new, 1)

out_py = REPO / "train_real.py"
out_py.write_text(src, encoding="utf-8")
print(f"已生成: {out_py}")

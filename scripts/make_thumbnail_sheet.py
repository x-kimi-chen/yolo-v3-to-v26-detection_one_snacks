# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""生成所有图片的缩略图 contact sheet，快速浏览全数据集"""
import os, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
SRC = ROOT / "my_images"
OUT = ROOT / "autolabel"
OUT.mkdir(exist_ok=True)

files = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png")))
CELL = 210
COLS = 13
rows = math.ceil(len(files) / COLS)

sheet = Image.new("RGB", (COLS * CELL, rows * CELL), (240, 240, 240))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for i, f in enumerate(files):
    im = Image.open(SRC / f)
    im.thumbnail((CELL - 8, CELL - 8))
    x = (i % COLS) * CELL + (CELL - im.width) // 2
    y = (i // COLS) * CELL + (CELL - im.height) // 2
    sheet.paste(im, (x, y))
    # 序号标注
    draw.rectangle([(i % COLS) * CELL, (i // COLS) * CELL, (i % COLS) * CELL + 34, (i // COLS) * CELL + 22], fill=(0, 0, 0))
    draw.text(((i % COLS) * CELL + 3, (i // COLS) * CELL + 1), str(i), fill=(255, 255, 0), font=font)

out_path = OUT / "all_thumbnails.jpg"
sheet.save(out_path, quality=82)
print(f"已生成: {out_path}  ({sheet.width}x{sheet.height}, {len(files)}张)")
# 输出序号-文件名对照（方便后面对照）
with open(OUT / "thumbnail_index.txt", "w", encoding="utf-8") as fp:
    for i, f in enumerate(files):
        fp.write(f"{i}\t{f}\n")
print("序号对照: autolabel/thumbnail_index.txt")

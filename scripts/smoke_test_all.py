# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
全版本冒烟测试：验证 ultralytics 原生支持的各个 YOLO 版本能否在本机 GPU 上完成训练
用 coco8 迷你数据集，每个版本跑 1 epoch，重点是"跑通"而非效果。
"""
import sys, os, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
SMOKE = ROOT / "datasets/_smoke"
RUNS = ROOT / "runs_smoke"

# 生成 coco8 本地配置（基于 ultralytics 内置配置，仅改 path）
import yaml
from ultralytics import __file__ as ul_file
coco8_src = Path(ul_file).parent / "cfg/datasets/coco8.yaml"
with open(coco8_src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["path"] = str((SMOKE / "coco8").as_posix())
coco8_yaml = SMOKE / "coco8_local.yaml"
with open(coco8_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

# 待测版本：名称 -> 模型（.pt = 预训练权重, .yaml = 从零训练）
MODELS = [
    ("v3",  "yolov3.yaml"),
    ("v5",  "yolov5nu.pt"),
    ("v6",  "yolov6.yaml"),
    ("v8",  "yolov8n.pt"),
    ("v9",  "yolov9t.yaml"),
    ("v10", "yolov10n.yaml"),
    ("v11", "yolo11n.pt"),
    ("v12", "yolo12n.yaml"),
    ("26",  "yolo26n.pt"),
]

def main():
    from ultralytics import YOLO
    results = []
    for tag, model_file in MODELS:
        print(f"\n{'='*60}\n[{tag}] 开始: {model_file}\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            model = YOLO(model_file)
            model.train(
                data=str(coco8_yaml),
                epochs=1, imgsz=320, batch=4, workers=2,
                device=0, project=str(RUNS), name=f"{tag}_smoke",
                exist_ok=True, plots=False, verbose=False,
                cache=False,
            )
            dt = time.time() - t0
            results.append((tag, model_file, "OK", f"{dt:.0f}s"))
            print(f"[{tag}] 成功 ({dt:.0f}s)", flush=True)
        except Exception as e:
            dt = time.time() - t0
            results.append((tag, model_file, f"失败: {type(e).__name__}", f"{dt:.0f}s"))
            print(f"[{tag}] 失败: {e}", flush=True)
            traceback.print_exc()

    print(f"\n\n{'='*60}\n冒烟测试汇总\n{'='*60}")
    for tag, mf, status, dt in results:
        mark = "PASS" if status == "OK" else "FAIL"
        print(f"[{mark}] {tag:4s} {mf:16s} {status} {dt}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""汇总 runs_train 下所有版本的结果 -> runs_train/_RESULTS.md
优先用各训练日志的最终验证数值；v4 用独立评估 (v4_eval.json)"""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
TRAIN = ROOT / "runs_train"
ORDER = ["yoloV3", "yoloV4", "yoloV5", "yoloV6", "yoloV7", "yoloV8",
         "yoloV9", "yoloV10", "yoloV11", "yoloV12", "yoloV13", "yoloV26"]

ALL_LINE = re.compile(r"\ball\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?")

def parse_log_final(name):
    """从日志末尾抓最终 'all' 行: 支持 ultralytics 与 v13（v13 会多一个 mAP75 列）"""
    for log_name in (f"{name}.log",):
        f = TRAIN / log_name
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
        matches = ALL_LINE.findall(txt)
        if matches:
            m = matches[-1]
            # m: (images, instances, P, R, mAP50, 第4个数, 可选第5个数)
            # v13 fork 会输出 mAP75 列（共5个数），此时 mAP50-95 取第5个
            map5095 = m[6] if m[6] else m[5]
            return {"P": m[2], "R": m[3], "mAP50": m[4], "mAP5095": map5095, "note": "最终验证"}
    return None

def parse_csv(name):
    f = TRAIN / name / "results.csv"
    if not f.exists():
        return None
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows:
            return None
        last = rows[-1]
        def col(*names):
            for n in names:
                for k in last:
                    if k.strip() == n:
                        return last[k]
            return "-"
        return {"epochs": col("epoch"), "P": col("metrics/precision(B)"),
                "R": col("metrics/recall(B)"), "mAP50": col("metrics/mAP50(B)"),
                "mAP5095": col("metrics/mAP50-95(B)"), "note": "csv末行"}
    except Exception as e:
        return {"note": f"解析失败 {e}"}

def parse_v7():
    f = TRAIN / "yoloV7" / "results.txt"
    if not f.exists():
        return None
    # v7 行格式: 99/99 gpu_mem box obj cls total labels imgsz | P R mAP50 mAP5095 val_losses...
    row_re = re.compile(r"^\s*\d+/\d+\s+\S+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    last = None
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = row_re.match(line)
        if m:
            last = m
    if last:
        return {"P": last.group(1), "R": last.group(2), "mAP50": last.group(3),
                "mAP5095": last.group(4), "note": "results.txt末行"}
    return None

def parse_v4():
    f = TRAIN / "v4_eval.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return {"P": "-", "R": "-", "mAP50": str(d.get("mAP50", "-")),
                "mAP5095": str(d.get("mAP5095", "-")), "note": "独立评估"}
    except Exception:
        return None

rows = []
for name in ORDER:
    if name == "yoloV7":
        r = parse_v7() or parse_csv(name)
    elif name == "yoloV4":
        r = parse_v4()
    else:
        r = parse_log_final(name) or parse_csv(name)
    if not r:
        r = parse_csv(name)
    if r:
        # epochs 信息补充
        if "epochs" not in r:
            c = parse_csv(name)
            if c and c.get("epochs"):
                r["epochs"] = c["epochs"]
        rows.append((name, r))

out = ["# 全版本训练结果汇总\n",
       "统一设置: imgsz=640, epochs=100, 同一 train/val 划分 (137/34), 同机同卡\n",
       "| 版本 | epochs | P | R | mAP@0.5 | mAP@0.5-0.95 | 数据来源 |",
       "|------|--------|---|---|---------|--------------|----------|"]
for name, r in rows:
    out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
        name, r.get("epochs", "-"), r.get("P", "-"), r.get("R", "-"),
        r.get("mAP50", "-"), r.get("mAP5095", "-"), r.get("note", "")))
missing = [n for n in ORDER if n not in [x[0] for x in rows]]
if missing:
    out.append(f"\n未完成/未解析: {', '.join(missing)}")
(TRAIN / "_RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("汇总完成 ->", TRAIN / "_RESULTS.md")
for name, r in rows:
    print(f"  {name}: mAP50={r.get('mAP50', '-')} ({r.get('note', '')})")

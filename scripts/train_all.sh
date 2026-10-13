#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
# ============================================================
# YOLO 全版本训练队列（顺序执行，可断点续跑：已完成版本自动跳过）
# 顺序策略：轻量快版先行，重量级(v7/v4/v3)压轴
# ============================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # 仓库根目录（本脚本位于 <repo>/scripts/ 下）
PY="${YOLO_PY:-python}"      # 如 python 不在 PATH，可用 YOLO_PY=/你的/python 指定
PY13="${YOLO_PY13:-python}"  # YOLOv13 需在专用环境执行（见 README）
DATA="$ROOT/dataset/inspection"
TRAIN="$ROOT/runs_train"
LOG="$TRAIN/_queue.log"
mkdir -p "$TRAIN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
clean_cache(){ find "$DATA" -name "*.cache" -delete 2>/dev/null; }

run_ultra(){  # $1=名称 $2=模型 $3=batch
  local name=$1 model=$2 batch=$3
  if [ -f "$TRAIN/$name/weights/best.pt" ]; then log "SKIP $name (已完成)"; return; fi
  clean_cache
  log "START $name model=$model batch=$batch"
  cd "$ROOT" && "$PY" scripts/_run_ultra.py "$name" "$model" "$batch" >> "$TRAIN/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  clean_cache
}

log "========== 训练队列启动 =========="

# ---- 第一批：ultralytics 轻量版本（每个十几分钟） ----
run_ultra yoloV26 yolo26n.pt 8
run_ultra yoloV8  yolov8n.pt 8
run_ultra yoloV11 yolo11n.pt 8
run_ultra yoloV12 yolo12n.pt 8
run_ultra yoloV10 yolov10n.pt 8
run_ultra yoloV9  yolov9t.pt 8
run_ultra yoloV5  yolov5nu.pt 8
run_ultra yoloV6  yolov6.yaml 8

# ---- YOLOv13（独立环境） ----
if [ ! -f "$TRAIN/yoloV13/weights/best.pt" ]; then
  clean_cache
  log "START yoloV13"
  cd "$ROOT/yolo-repos/yolov13" && "$PY13" "$ROOT/scripts/_run_ultra13.py" >> "$TRAIN/yoloV13.log" 2>&1
  log "END yoloV13 rc=$?"
  clean_cache
else log "SKIP yoloV13 (已完成)"; fi

# ---- YOLOv7（老仓库，workers=0） ----
if [ ! -f "$TRAIN/yoloV7/weights/best.pt" ]; then
  clean_cache
  log "START yoloV7"
  cd "$ROOT/yolo-repos/yolov7" && "$PY" train.py \
    --weights yolov7.pt --cfg cfg/training/yolov7.yaml --data "$DATA/v7_data.yaml" \
    --hyp data/hyp.scratch.p5.yaml --batch-size 4 --epochs 100 --img-size 640 640 \
    --device 0 --workers 0 --project "$TRAIN" --name yoloV7 --exist-ok >> "$TRAIN/yoloV7.log" 2>&1
  log "END yoloV7 rc=$?"
  clean_cache
else log "SKIP yoloV7 (已完成)"; fi

# ---- YOLOv4（bubbliiiing，从零训练） ----
if [ ! -f "$TRAIN/yoloV4/best_epoch_weights.pth" ]; then
  clean_cache
  log "START yoloV4"
  cd "$ROOT/yolo-repos/yolov4-pytorch" && "$PY" train_real.py >> "$TRAIN/yoloV4.log" 2>&1
  log "END yoloV4 rc=$?"
  clean_cache
else log "SKIP yoloV4 (已完成)"; fi

# ---- YOLOv3（最大模型，压轴） ----
run_ultra yoloV3 yolov3u.pt 4

log "========== 全部完成 =========="
cd "$ROOT" && "$PY" scripts/collect_results.py >> "$LOG" 2>&1
log "结果汇总已生成: runs_train/_RESULTS.md"

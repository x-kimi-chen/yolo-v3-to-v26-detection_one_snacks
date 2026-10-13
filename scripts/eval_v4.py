# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 x-kimi-chen
"""
v4 独立评估：用 bubbliiiing 的解码器 + 自写 mAP，在验证集上评估 best 权重
结果写入 runs_train/v4_eval.json
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本脚本位于 <repo>/scripts/ 下）
REPO = ROOT / "yolo-repos/yolov4-pytorch"
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image


def main():
    from nets.yolo import YoloBody
    from utils.utils_bbox import DecodeBox
    from utils.utils import cvtColor, get_classes, get_anchors, preprocess_input, resize_image

    input_shape = [640, 640]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes_path = str(ROOT / "classes.txt")
    anchors_path = str(REPO / "model_data/yolo_anchors.txt")
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = YoloBody(anchors_mask, num_classes, pretrained=False)
    best = ROOT / "runs_train/yoloV4/best_epoch_weights.pth"
    state = torch.load(str(best), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and not any("." in k for k in list(state.keys())[:1]):
        state = state["model"]
    model.load_state_dict(state)
    model = model.cuda().eval()

    bbox_util = DecodeBox(anchors, num_classes, input_shape, anchors_mask)

    val_img = ROOT / "dataset/inspection/images/val"
    val_lbl = ROOT / "dataset/inspection/labels/val"
    iou_ths = np.arange(0.5, 1.0, 0.05)

    # 收集预测与GT
    all_preds = {t: {c: [] for c in range(num_classes)} for t in range(len(iou_ths))}
    all_gt = {c: [] for c in range(num_classes)}

    img_files = sorted(list(val_img.glob("*.jpg")) + list(val_img.glob("*.png")))
    for img_path in img_files:
        image = Image.open(img_path)
        image = cvtColor(image)
        iw, ih = image.size
        img_data = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image=True)
        img_data = preprocess_input(np.array(img_data, np.float32))
        img_data = np.transpose(img_data, (2, 0, 1))[None]
        with torch.no_grad():
            out = model(torch.from_numpy(img_data).cuda())
            out = bbox_util.decode_box(out)
            results = bbox_util.non_max_suppression(torch.cat(out, 1), num_classes, input_shape,
                                                    (ih, iw), True, conf_thres=0.001, nms_thres=0.65)
        det = results[0]
        if hasattr(det, "cpu"):
            det = det.cpu().numpy()
        det = np.asarray(det, dtype=np.float64)
        # bubbliiiing 的 NMS 输出为 7 列: [y1, x1, y2, x2, obj_conf, cls_conf, cls_id]（注意x/y转置！）
        for d in det:
            cls_id = int(round(d[6]))
            conf = float(d[4]) * float(d[5])
            x1 = max(0.0, min(float(d[1]), iw))
            y1 = max(0.0, min(float(d[0]), ih))
            x2 = max(0.0, min(float(d[3]), iw))
            y2 = max(0.0, min(float(d[2]), ih))
            if 0 <= cls_id < num_classes and x2 > x1 and y2 > y1:
                all_preds[0][cls_id].append((img_path.name, x1, y1, x2, y2, conf))

        lbl = val_lbl / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = map(float, line.split()[:5])
                x1, y1 = (cx - w / 2) * iw, (cy - h / 2) * ih
                x2, y2 = (cx + w / 2) * iw, (cy + h / 2) * ih
                all_gt[int(c)].append((img_path.name, x1, y1, x2, y2))

    print(f"验证图片 {len(img_files)} 张, GT框 {sum(len(v) for v in all_gt.values())}, 预测框 {sum(len(v) for v in all_preds[0].values())}")

    def iou_xyxy(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter + 1e-9)

    def ap_at(preds_c, gt_c, thr):
        if len(gt_c) == 0:
            return None
        if len(preds_c) == 0:
            return 0.0
        preds = sorted(preds_c, key=lambda p: -p[5])
        gt_used = {i: False for i in range(len(gt_c))}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for pi, p in enumerate(preds):
            best_iou, best_gi = 0, -1
            for gi, g in enumerate(gt_c):
                if g[0] != p[0] or gt_used[gi]:
                    continue
                v = iou_xyxy(p[1:5], g[1:5])
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= thr:
                tp[pi] = 1
                gt_used[best_gi] = True
            else:
                fp[pi] = 1
        tpc = np.cumsum(tp)
        fpc = np.cumsum(fp)
        rec = tpc / len(gt_c)
        prec = tpc / (tpc + fpc + 1e-9)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    result = {"mAP50": None, "mAP5095": None, "per_class_50": {}}
    aps50, aps_all = [], []
    for c in range(num_classes):
        ap50 = ap_at(all_preds[0][c], all_gt[c], 0.5)
        if ap50 is None:
            continue
        aps50.append(ap50)
        result["per_class_50"][class_names[c]] = round(ap50, 4)
        # 50-95 平均
        vals = [ap_at(all_preds[0][c], all_gt[c], t) for t in iou_ths]
        vals = [v for v in vals if v is not None]
        aps_all.append(np.mean(vals))
    result["mAP50"] = round(float(np.mean(aps50)), 5)
    result["mAP5095"] = round(float(np.mean(aps_all)), 5)
    print("v4 评估结果:", json.dumps(result, ensure_ascii=False, indent=1))
    (ROOT / "runs_train/v4_eval.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print("已保存 runs_train/v4_eval.json")


if __name__ == "__main__":
    main()

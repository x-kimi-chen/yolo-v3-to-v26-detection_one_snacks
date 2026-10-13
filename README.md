# YOLO 全版本目标检测项目（v3 → v26）

**在一台 RTX 5060 笔记本上，完成从 YOLOv3 到 YOLO26 共 12 个版本的完整训练、评测与横向对比——12/12 全部跑通并附训练权重。**

零食产品视觉检测 · 5 类目标 · 171 张自建数据集 · AGPL-3.0 开源

> ⚠️ **本项目仅供学习研究与技术交流使用，请勿用于商业用途。** 详见文末「用途声明」。

> 🚀 **第一次来？** 先看 [5 分钟演示导览](docs/项目演示说明.md)（30 秒了解项目 → 成果速览 → 亲手体验）。

## ✨ 项目亮点

- 🎯 **全版本覆盖**：YOLOv3 / v4 / v5 / v6 / v7 / v8 / v9 / v10 / v11 / v12 / v13 / YOLO26，共 12 个版本，同数据、同评测口径横向对比
- 📊 **训练成果**：验证集 mAP@0.5 最高 **0.983**（YOLOv12 / YOLO26 并列），10 个版本达到 0.977 以上
- 🏷️ **自建数据集**：171 张实拍图片（55 个连拍组）、5 类商品、815 个标注框；按连拍组划分 train/val，杜绝近重复图片导致的数据泄漏
- 🤖 **自动预标注流水线**：FastSAM 分割提案 + 颜色相似度聚类分类 + 自动补救，标注效率提升约 80%
- 📖 **完整中文文档**：使用说明（Word 详细版）、训练指南、GitHub 上传说明、10+ 条实战踩坑 FAQ
- 🔧 **工程化脚本**：一键训练队列（支持断点续训）、同图多版本推理对比、独立评估工具

## 📈 成绩总览

统一设置：imgsz=640、epochs=100、同一 train/val 划分（137/34）、同一硬件。

| 版本 | mAP@0.5 | mAP@0.5-0.95 | 权重文件 | 备注 |
|------|---------|--------------|----------|------|
| **YOLOv12** | **0.983** | 0.961 | weights/yoloV12_best.pt | 🥇 推荐 |
| **YOLO26** | **0.983** | 0.953 | weights/yoloV26_best.pt | 🥇 推荐 |
| YOLOv3 | 0.982 | 0.956 | Release 附件 | |
| YOLOv8 | 0.980 | 0.955 | weights/yoloV8_best.pt | |
| YOLO11 | 0.980 | 0.952 | weights/yoloV11_best.pt | |
| YOLOv5 | 0.979 | 0.957 | weights/yoloV5_best.pt | |
| YOLOv13 | 0.979 | 0.957 | weights/yoloV13_best.pt | |
| YOLOv7 | 0.978 | 0.935 | weights/yoloV7_best.pt | |
| YOLOv10 | 0.978 | 0.955 | weights/yoloV10_best.pt | |
| YOLOv9 | 0.977 | 0.955 | weights/yoloV9_best.pt | |
| YOLOv6 | 0.966 | 0.882 | weights/yoloV6_best.pt | 从零训练 |
| YOLOv4 | 0.944 | 0.686 | Release 附件 | 从零训练 |

> YOLOv6 / YOLOv4 官方无预训练权重，属从零训练；YOLOv3 与 YOLOv4 权重超过 GitHub 100MB 单文件限制，放在 [Releases](../../releases) 附件中。

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+（本项目在 3.14 上开发验证）
- NVIDIA GPU（建议 6GB+ 显存；CPU 也可推理）
- 依赖：PyTorch + ultralytics（`requirements.txt` 见下）

### 2. 安装

```bash
# 1) 安装 PyTorch（按你的 CUDA 版本选择，示例为 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) 安装 ultralytics
pip install ultralytics

# 3)（可选）老框架版本 v4/v7/v13 需要单独克隆对应仓库
git clone https://github.com/WongKinYiu/yolov7.git            # GPL-3.0
git clone https://github.com/bubbliiiing/yolov4-pytorch.git   # MIT
git clone https://github.com/iMoonLab/yolov13.git             # AGPL-3.0
```

### 3. 推理（推荐 YOLOv12 / YOLO26）

```python
from ultralytics import YOLO

model = YOLO("weights/yoloV12_best.pt")
results = model.predict("your_image.jpg", imgsz=640, conf=0.25, device=0)

for box in results[0].boxes:
    print(model.names[int(box.cls[0])], float(box.conf[0]), box.xyxy[0].tolist())

results[0].save(filename="result.jpg")   # 保存带框结果图
```

批量推理：把 `predict()` 的输入换成图片文件夹路径即可。

### 4. 类别定义

| 索引 | 类名 | 含义 |
|------|------|------|
| 0 | puff_vanilla | 泡芙·香草味（蓝色袋） |
| 1 | puff_strawberry | 泡芙·草莓味（粉色袋） |
| 2 | gluten_spicy | 素毛肚·香辣味（红色袋） |
| 3 | gluten_pepper | 素毛肚·山椒味（绿色袋） |
| 4 | cracker | 黑松露火腿苏打饼干（金色盒） |

## 📦 权重下载

- **仓库内**（克隆即可用）：`weights/` 目录包含 10 个版本的训练权重
- **完整 12 版本权重包**（含超过 GitHub 单文件限制的 YOLOv3 / YOLOv4）：见 [Releases](../../releases) 页面的 `yolo-v3-v26-12versions-weights.zip`

```python
from ultralytics import YOLO

model = YOLO("weights/yoloV12_best.pt")                 # 或解压后的任意版本权重
model.predict("your_image.jpg", imgsz=640, save=True)   # 一行完成推理
```

## 🗂️ 项目结构

```
.
├── README.md                 本文件
├── LICENSE                   AGPL-3.0 许可证
├── THIRD_PARTY_NOTICES.md    第三方开源组件协议标注
├── classes.txt               类别定义
├── scripts/                  全部功能脚本（训练/评估/自动标注/对比）
├── dataset/inspection/       数据集（train 137 / val 34，YOLO 格式）
├── weights/                  10 个版本的训练权重（小文件）
├── results/                  成绩单、训练曲线、12版本推理对比图
├── autolabel/prelabels/      自动预标注结果（171 个 YOLO 标签）
└── docs/
    ├── 项目演示说明.md                         访客导览与现场演示脚本（先看这个）
    ├── YOLO全版本检测项目使用说明_详细版.docx   完整使用手册
    └── GitHub上传说明.md                       本仓库的上传/维护指引
```

## 🧠 训练（速览）

```bash
# 单版本微调（以 v12 为例）
python scripts/_run_ultra.py yoloV12 yolo12n.pt 8

# 全队列训练（12 个版本依次训练，断点续训）
bash scripts/train_all.sh
```

详细的数据准备、标注流程、各版本注意事项，见 `docs/YOLO全版本检测项目使用说明_详细版.docx`。

> 仓库内脚本已做**路径解耦**：基于 `scripts/` 位置自动识别项目根目录，克隆到任意目录即可使用；若 Python 不在 PATH 中，训练队列可用 `YOLO_PY=/你的python路径 bash scripts/train_all.sh` 指定解释器。

## ⚠️ 用途声明

1. 本项目为个人学习研究作品，**仅供学习、研究与技术交流使用，不得用于商业用途**；
2. 未经授权，不得将本项目（含数据集照片、训练权重）用于任何生产环境、商品结算、商业宣传等场景；
3. 数据集照片中的商品包装与商标归各自权利人所有，作者不主张相关权利，亦不构成任何形式的品牌授权或背书；
4. 如需将相关技术用于商业场景：请就 Ultralytics（AGPL-3.0）申请企业许可，并就商品照片等素材另行取得权利人授权；
5. 作者不对使用本项目造成的任何后果承担责任。

## 📄 许可证与致谢

**Copyright (C) 2026 x-kimi-chen** · 本项目以 **AGPL-3.0-or-later** 许可证开源（见 [LICENSE](LICENSE)），源码文件均带有 SPDX 标识。

本项目为作者独立开发的个人开源项目（非 Ultralytics 官方项目），与图片中涉及的任何品牌方无关联。

> ⚠️ 重要：本项目使用 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）完成主要版本的训练与推理。根据 Ultralytics 官方要求，使用其代码/模型的项目应以 AGPL-3.0 开源。请遵守对应条款，商业闭源使用请联系 Ultralytics 获取企业许可。

各第三方组件与资源的完整协议、来源与版权标注见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（v3u/v5u/v6/v8/v9/v10/v11/v12/v26 训练框架）
- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)（v7 框架）
- [bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)（v4 框架）
- [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)（v13 框架）
- [CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)（数据标注工具）

> 数据集中的商品照片由本项目作者（x-kimi-chen）拍摄，图片中出现的商品包装、商标归各自权利人所有，仅用于技术研究示例，不构成任何商业背书。

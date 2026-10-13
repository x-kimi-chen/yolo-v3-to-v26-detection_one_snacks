# 第三方开源组件协议标注（THIRD PARTY NOTICES）

本项目（YOLO 全版本目标检测项目）在开发与训练过程中使用了以下第三方开源组件。各组件版权归其原作者所有，遵循其各自的开源许可证。本文件按组件用途分类标注。

---

## 1. 训练/推理框架（核心依赖）

| 组件 | 许可证 | 版权 | 用途 | 链接 |
|------|--------|------|------|------|
| Ultralytics YOLO | **AGPL-3.0** | © Ultralytics Inc. | v3u/v5u/v6/v8/v9/v10/v11/v12/v26 及 v13 分支的训练与推理框架 | https://github.com/ultralytics/ultralytics |
| WongKinYiu/yolov7 | **GPL-3.0** | © WongKinYiu 等 | YOLOv7 训练与推理 | https://github.com/WongKinYiu/yolov7 |
| bubbliiiing/yolov4-pytorch | **MIT** | © 2020 JiaQi Xu | YOLOv4 训练与推理 | https://github.com/bubbliiiing/yolov4-pytorch |
| iMoonLab/yolov13 | **AGPL-3.0** | © iMoonLab | YOLOv13 训练与推理（Ultralytics 分支） | https://github.com/iMoonLab/yolov13 |

> ⚠️ **重要说明**：上述框架**源码未包含在本仓库中**（通过 pip 安装或自行克隆）。本项目仅包含基于这些框架编写的训练脚本与训练产物（模型权重）。由于核心框架使用 AGPL-3.0（Ultralytics 官方要求使用其代码/模型的项目以 AGPL-3.0 开源），**本仓库整体以 AGPL-3.0 发布**。

## 2. 数据标注工具

| 组件 | 许可证 | 版权 | 用途 | 链接 |
|------|--------|------|------|------|
| X-AnyLabeling | **GPL-3.0** | © CVHub / Wei Wang | 数据标注与预标注审核工具 | https://github.com/CVHub520/X-AnyLabeling |

> 本项目使用 X-AnyLabeling 完成数据标注，但**未重新分发该软件**（软件本体请从其官方渠道获取）。

## 3. Python 运行环境与基础库

| 组件 | 许可证 | 用途 |
|------|--------|------|
| PyTorch / TorchVision | BSD-3-Clause | 深度学习框架 |
| NumPy | BSD-3-Clause | 数值计算 |
| OpenCV (opencv-python) | Apache-2.0 | 图像处理 |
| Pillow | HPND (PIL License) | 图像读取 |
| SciPy | BSD-3-Clause | 科学计算 |
| Matplotlib | PSF-based (Matplotlib License) | 绘图 |
| PyYAML | MIT | 配置文件解析 |
| tqdm | MIT / MPL-2.0 | 进度条 |
| pandas | BSD-3-Clause | 数据处理 |
| psutil | BSD-3-Clause | 系统监控 |

> 以上库通过 pip 安装使用，未修改、未重新分发其源码。

## 4. 预训练权重说明

本项目训练时使用过各框架官方的 **COCO 预训练权重**（如 yolov8n.pt、yolo12n.pt 等）作为起点，这些文件：

- **未包含在本仓库中**（已在 .gitignore 排除，请从各官方渠道自动/手动获取）
- 其版权与许可随各自框架（Ultralytics 资产为 AGPL-3.0；yolov7.pt 为 GPL-3.0）

本仓库 **`weights/` 目录中的模型权重**均为**本项目作者使用自有数据训练产生**的成果，按本仓库 AGPL-3.0 许可证发布。

## 5. 数据集声明

`dataset/` 与 `autolabel/prelabels/` 中的图片与标注为**本项目作者自建**：

- 图片为作者自行拍摄的商品实拍图
- 标注为作者/自动预标注 + 人工修正产生
- 图片中出现的商品包装设计、品牌商标归各自权利人所有；本数据集仅用于技术研究与教学示例，不代表与相关品牌的任何关联或背书
- 如权利人认为使用不当，请联系移除

## 6. 合规要点小结

1. **本仓库整体采用 AGPL-3.0**，符合 Ultralytics 官方对使用其代码/模型项目的开源要求；
2. 第三方框架源码、官方预训练权重、标注工具软件均**未打包进仓库**，仅通过安装说明引用；
3. 所有使用到的组件均已在本文档中标注许可证与版权归属；
4. 如需**商业闭源使用**本项目：须就 Ultralytics 部分向 Ultralytics 申请企业许可（Enterprise License），并遵守其余组件的对应条款。

---

*本文件仅供工程参考，不构成法律意见；正式商用前建议咨询专业法律顾问。*

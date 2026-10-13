# GitHub 上传说明（部署指引）

> 本文件手把手教你把这个项目发布到 GitHub，包括：准备工作 → 建仓库 → 上传代码 → 发布权重 → 日常维护。全程约 15 分钟。
>
> 💡 **使用 GitHub Desktop 的用户请直接看第八章**（图形化流程、无需命令行，可替代第二、三章）。

---

## 〇、发布包说明

发布内容已整理在 **`github_ready/`** 文件夹中（本文件即位于其中），结构如下：

| 内容 | 说明 | 是否随仓库上传 |
|------|------|----------------|
| README.md | 项目详细说明（GitHub 首页展示） | ✅ |
| LICENSE | AGPL-3.0 许可证全文（合规必需） | ✅ |
| THIRD_PARTY_NOTICES.md | 第三方开源组件协议标注 | ✅ |
| .gitignore | 排除规则（防止把大文件/第三方内容传上去） | ✅ |
| classes.txt | 类别定义 | ✅ |
| scripts/ | 全部功能脚本 | ✅ |
| dataset/inspection/ | 数据集（137 训练 + 34 验证，53MB） | ✅ |
| weights/ | 10 个版本训练权重（123MB，单文件均 < 100MB） | ✅ |
| results/ | 成绩单、训练曲线、12 版本推理对比图 | ✅ |
| autolabel/prelabels/ | 自动预标注结果（171 个标签文件） | ✅ |
| docs/ | 使用说明（详细版）等文档 | ✅ |

**未包含（且不应上传）的内容**：第三方框架源码（`yolo-repos/`）、标注工具（`tools/` 571MB）、官方 COCO 预训练权重（根目录 `*.pt`）、训练运行目录（`runs_train/`）、原始高清图片（`my_images/`）。这些已在 `.gitignore` 中排除——**原因：第三方代码/权重版权与体积限制，详见 `THIRD_PARTY_NOTICES.md`**。

**两个超限权重**：`yoloV3_best.pt`（198MB）和 `yoloV4_best.pth`（244MB）超过 GitHub 单文件 100MB 限制，**不直接上传**，改为作为 Release 附件发布（见第三步）。完整 12 版本权重包已打好：`output/yolo-v3-v26-12versions-weights.zip`（565MB）。

---

## 一、准备工作

1. **注册/登录 GitHub**：https://github.com
2. **安装 Git**：https://git-scm.com/downloads （Windows 一路默认安装即可）
3. **确认版本信息**（首次使用需配置一次）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

4. **网络提示（国内环境）**：GitHub 直连不稳定时，建议开启代理后再执行推送（见文末常见问题⑶）。

---

## 二、第一步：在 GitHub 创建空仓库

1. 打开 https://github.com/new
2. 填写：
   - **Repository name**：`yolo-v3-to-v26-detection_one_snacks`（或你喜欢的名字，建议英文）
   - **Description**：`YOLOv3 → YOLO26 全版本零食产品检测：12 个版本训练、对比与使用指南`
   - 可见性：**Public**（公开）或 Private（私有）按需选择
3. ⚠️ **不要勾选** "Add a README file"、"Add .gitignore"、"Choose a license"（发布包里已有这些文件，勾了会冲突）
4. 点击 **Create repository**，记下页面显示的仓库地址（形如 `https://github.com/你的用户名/yolo-v3-to-v26-detection_one_snacks.git`）

---

## 三、第二步：上传本地发布包

打开命令行（Git Bash / CMD / PowerShell 均可），执行：

```bash
# 1. 进入发布包目录（注意替换为你机器上的实际路径）
cd D:/yolo-project_one/github_ready

# 2. 初始化仓库并提交
git init -b main
git add .
git commit -m "初始发布：YOLOv3→YOLO26 全版本检测项目（12版本训练+数据集+权重+文档）"

# 3. 关联远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/yolo-v3-to-v26-detection_one_snacks.git

# 4. 推送
git push -u origin main
```

推送成功后，刷新 GitHub 页面即可看到项目。

> **体积检查**：本次上传约 180MB（含 123MB 权重 + 53MB 数据集），单文件最大 71MB，均在 GitHub 限制内（单文件 <100MB，仓库 <1GB 建议值）。
>
> **首次推送会要求登录**：推荐使用 **GitHub Desktop**（图形化，自动处理登录）或 **Personal Access Token**（见常见问题⑴）。

---

## 四、第三步：发布大权重（Releases）

`yoloV3_best.pt`（198MB）与 `yoloV4_best.pth`（244MB）超过单文件限制，通过 Releases 发布：

1. 进入你的仓库页面 → 右侧 **Releases** → **Create a new release**（或直接访问 `你的仓库地址/releases/new`）
2. 填写：
   - **Choose a tag**：输入 `v1.0.0` → 点击 "Create new tag"
   - **Release title**：`v1.0.0 - 12 版本训练权重`
   - **描述**：可写 `包含全部 12 个版本训练权重（含超过 100MB 的 YOLOv3/v4），以及完整权重包 zip`
3. 上传附件（拖拽即可）：
   - `output/yolo-v3-v26-12versions-weights.zip`（565MB，完整 12 版本权重包，推荐）
   - 如需单文件：另附 `yoloV3_best.pt` 和 `yoloV4_best.pth`（各 <2GB 均支持）
4. 点击 **Publish release**

完成后，README 中的权重下载说明可直接链接到该 Release 页面。

---

## 五、第四步：完善仓库展示（可选但推荐）

1. **About 区**（仓库首页右上齿轮）：
   - Description：填写项目简介
   - Topics（标签）：`yolo` `object-detection` `yolov8` `yolov12` `ultralytics` `pytorch` `deep-learning`
2. 主页预览：README.md 会自动渲染为首页，检查图片与表格是否正常。

---

## 六、常见问题

**⑴ 推送时提示认证失败 / 要求密码？**
GitHub 已不支持账号密码推送，三选一：
- 推荐：安装 **GitHub Desktop**（官网下载），登录后用它推送（`Add local repository` 选择 `github_ready` 文件夹即可）
- 使用 **Personal Access Token**：GitHub → Settings → Developer settings → Personal access tokens → 生成 token（勾选 `repo` 权限），推送时用户名填 GitHub 用户名、密码粘贴 token
- 配置 **SSH Key**：生成 ssh key 后添加到 GitHub（适合长期使用）

**⑵ 推送报错 "file is 198.00 MB; this exceeds GitHub's file size limit of 100.00 MB"？**
说明有大文件混入。检查 `.gitignore` 是否生效（确认 `weights/yoloV3_best.pt`、`weights/yoloV4_best.pth` 未被加入），必要时执行：
```bash
git rm --cached <大文件路径>   # 从暂存区移除（保留本地文件）
git commit -m "移除超限文件"
```

**⑶ 推送超时 / 连接失败（国内网络）？**
开启代理后执行：
```bash
git config --global http.proxy http://127.0.0.1:你的代理端口
git config --global https.proxy http://127.0.0.1:你的代理端口
# 之后正常 push；不需要代理时可取消：
# git config --global --unset http.proxy && git config --global --unset https.proxy
```
备选方案：使用 Gitee（码云）镜像推送。

**⑷ 以后修改了内容，怎么更新？**
```bash
cd D:/yolo-project_one/github_ready
git add .
git commit -m "更新说明：修改了xxx"
git push
```

**⑸ 想重新整理发布内容？**
在 `D:/yolo-project_one/github_ready/` 中增删文件后按 ⑷ 重新提交即可。（注意：不要在发布包里放 `runs_train/`、`yolo-repos/`、`tools/` 等，原因见第〇节。）

---

## 七、发布前检查清单

- [ ] `github_ready/` 中无 `yolo-repos/`、`tools/`、`*.cache`、根目录 `*.pt` 等第三方/超限内容
- [ ] `LICENSE`（AGPL-3.0）与 `THIRD_PARTY_NOTICES.md` 已包含
- [ ] `weights/` 中单文件均 < 100MB（当前最大 71MB ✓）
- [ ] GitHub 仓库已创建且未勾选自动生成 README/LICENSE
- [ ] 推送成功后在网页确认 README 渲染正常
- [ ] Releases 中已上传完整权重包（含 v3/v4 大权重）
- [ ] 仓库 About 区填写了简介与 Topics

---

*如遇本说明未覆盖的问题，可在仓库 Issue 中记录，或回来让我协助排查。*

---

## 八、GitHub Desktop 操作流程（推荐路线）

> 本章替代"第二章 + 第三章"：用 GitHub Desktop 图形化完成建仓与上传，无需命令行、无需配置 Token。

### 1. 安装与登录

1. 下载安装 GitHub Desktop：https://desktop.github.com/
2. 打开后用你的 GitHub 账号登录（自动完成认证，后续推送无需再输密码）

### 2. 把发布包变成 Git 仓库

1. 菜单 `File → Add Local Repository…`
2. 选择目录：`D:\yolo-project_one\github_ready`
3. 若提示 "This directory does not appear to be a Git repository"，点击 **create a repository**（在该目录创建）
4. 在创建窗口中填写：
   - **Name**：`yolo-v3-to-v26-detection_one_snacks`（或你喜欢的名字）
   - **Description**：建议填 `YOLOv3→YOLO26 全版本零食产品检测：12 个版本全部跑通；本项目仅供研究学习使用`
   - **取消勾选** "Initialize with a README"（发布包里已有，勾了会冲突）
   - **Git Ignore**：选 `None`（发布包内已有 .gitignore）
   - **License**：选 `None`（发布包内已有 LICENSE）
5. 点击 **Create Repository**

### 3. 提交前的变更核对（重要）

左侧 **Changes** 列表会列出全部待提交文件（约 570 个）。逐项核对：

- ✅ 应出现：scripts/、dataset/、weights/、results/、docs/、autolabel/、README.md、LICENSE、THIRD_PARTY_NOTICES.md
- ❌ 不应出现：yolo-repos、tools、runs_train、my_images、根目录的官方预训练 *.pt、*.cache、.env 等
- 📦 最大文件应为 `weights/yoloV7_best.pt`（约 72MB，低于 GitHub 单文件 100MB 限制）

> 若弹出 "large files / Git LFS" 提示：本项目最大文件 72MB 未超普通限制，可**直接跳过 LFS**（选 Skip / No thanks）正常提交；如你希望规范管理大文件也可选择启用（克隆者需安装 Git LFS）。

### 4. 填写 Commit 并提交

底部 Summary 建议文案（可直接复制）：

```
初始发布：YOLOv3→YOLO26 全版本检测项目（12版本训练 · 数据集 · 权重 · 文档）
```

Description（可选）：

```
包含 12 个版本的训练脚本与权重、171 张自建数据集、自动预标注流水线、完整中文文档，以及 AGPL-3.0 许可与第三方协议标注。
```

点击 **Commit to main**。

### 5. 发布（公开推送）

点击顶部 **Publish repository** 按钮：

- 确认 Name / Description
- 公开仓库：**取消勾选 "Keep this code private"**
- 点击 **Publish Repository**

⏳ 上传约 179MB，视网速需要几分钟。

> ⚠️ **点下 Publish 即公开**——请先确认你已经完成第 3 步的文件核对，且第 4 步的 commit 已提交。

### 6. 发布大权重（Releases）

GitHub Desktop 不处理 Releases，请按本文**第四章**在网页操作（完整权重包 zip 已备好：`output/yolo-v3-v26-12versions-weights.zip`）。

### 7. 以后怎么更新

全程图形化：在 Desktop 中查看/修改文件 → 在 Changes 里填写 Summary → **Commit to main** → 点 **Push origin**。若网上有他人改动，先 **Fetch origin / Pull** 再推。

# PS_cutout image：科研图片精细抠图技能

这是面向 Codex 的科研图片精细抠图技能，重点处理黑色或灰黑背景下的水稻、盆栽、穗部、细叶、昆虫和显微主体。它会在原图同目录生成：

- `sample_cutout.psd`：包含纯黑背景层、透明抠图层、蒙版层和隐藏原图层的可编辑 PSD。
- `sample_clean_black.png`：背景为真正 RGB(0,0,0) 的最终 PNG。

原图不会被覆盖。Photoshop 不是自动流程的必需条件，但仍推荐用它进行 200%–400% 的最终边缘复核。

## 最省事的安装方法

### macOS：下载后双击

1. 在 GitHub 页面点击 **Code → Download ZIP**。
2. 解压 ZIP。
3. 双击 `install.command`。
4. 安装结束后重新打开 Codex，或新建一个 Codex 任务。

如果 macOS 首次阻止运行，请右键 `install.command`，选择“打开”。也可以在终端执行：

```bash
bash install.sh
```

### Windows

解压后，在仓库目录打开 PowerShell，执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### Linux

```bash
bash install.sh
```

安装器会自动完成以下工作：

1. 检查 Python 版本；支持 Python 3.11、3.12 和 3.13。
2. 把技能安装到 `~/.agents/skills/ps-cutout-image`。
3. 在用户目录创建独立虚拟环境，不修改系统 Python。
4. 安装 `requirements.txt` 中固定版本的依赖。
5. 预下载 U2Net CPU 模型，避免第一次抠图临时等待。
6. 生成 `runtime.json`，让技能自动找到正确的 Python。
7. 实际写入并重新解析一个测试 PSD，检查图层、透明 Alpha 和纯黑 RGB(0,0,0) 背景。

首次安装需要联网，并会下载数百 MB 的依赖和模型。此后核心流程可以使用本地模型运行。

## 从 GitHub 直接安装到 Codex

也可以使用通用 Skills 安装器：

```bash
npx --yes skills add https://github.com/huohuo143/ps-cutout-image-skill \
  --global --yes --copy

bash ~/.agents/skills/ps-cutout-image/install.sh
```

第二条命令用于安装隔离依赖、预下载模型并执行完整自检。完成后请新建 Codex 任务。

## 使用方法

在 Codex 中直接说：

> 使用 PS_cutout image 技能精细抠除这张水稻图片的黑布背景，输出纯黑背景 PNG 和可编辑 PSD，并检查叶尖、叶缝和盆边。

也可以直接运行：

```bash
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py /path/to/sample.JPG
```

常用模式：

```bash
# 自动判断场景
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py sample.JPG

# 小主体、橙色盆、大面积黑布
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py sample.JPG --segmentation-mode dark-rice

# 主体触及画面顶部或底部
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py sample.JPG --segmentation-mode dark-rice-frame

# 灰黑方盆、白色标牌、细长叶片、底部桌面
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py sample.JPG --segmentation-mode dark-rice-gray-pot
```

所有最终文件固定写入原图所在文件夹。`--output-dir` 即使传入也不会改变最终输出位置。

## 单独检查安装

```bash
python3 ~/.agents/skills/ps-cutout-image/run_cutout.py --help
```

完整环境自检：

```bash
~/.local/share/ps-cutout-image/.venv/bin/python \
  ~/.agents/skills/ps-cutout-image/scripts/verify_install.py --check-model
```

自检通过时输出 JSON 中的 `status` 为 `pass`。

## 依赖

依赖已固定在 `requirements.txt`：NumPy、Pillow、SciPy、OpenCV headless、psd-tools 和 `rembg[cpu]`。不需要 API key、云端账号或付费服务。

CPU 是默认且唯一的自动安装后端，用来避免不同显卡、CUDA 或 macOS CoreML 编译造成的不稳定。处理速度取决于图片尺寸和电脑性能。

## 质量边界

自动流程针对深色摄影背景的水稻与盆栽图进行了专门优化，但科研抠图仍然需要视觉复核。以下区域必须重点检查：

- 叶尖、交叉叶缝、苗心和盆口；
- 穗芒、根丝、昆虫足、触角和翅缘；
- 黑边、白边、蓝黑布残影和半透明雾边；
- 盆沿、盆底、白色标牌和底部桌面。

当自动 QA 给出 `manual_review_required` 或视觉质量不符合论文排版要求时，应在 Photoshop 中修正蒙版后重新运行收尾流程。

## 常见问题

### 提示 Python 版本不支持

安装 Python 3.11、3.12 或 3.13，然后重新运行安装器。当前固定依赖不面向 Python 3.10 及以下或 Python 3.14 及以上。

### 模型下载失败

检查网络后重新运行 `install.command`、`install.sh` 或 `install.ps1`。安装器会复用已经下载成功的文件。

### Codex 找不到技能

确认以下文件存在：

```text
~/.agents/skills/ps-cutout-image/SKILL.md
```

然后重新打开 Codex 或新建任务。

### 已有人工修好的 PSD

把 PSD 放在原图旁边，命名为 `sample_cutout.psd` 或 `sample.psd`。技能会优先读取其中带透明 Alpha 的抠图层，避免重新一键分割。

## 许可

本项目采用 MIT License。第三方依赖和模型遵循各自的许可证。

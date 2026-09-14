# 模型和公开数据资源

大权重与 demo **存本仓库 `data/foundationpose/`**，由脚本下载，**不提交 git**。

一键：

```powershell
python scripts\download_assets.py
python scripts\link_foundationpose_assets.py
```

详见 [DOWNLOAD_CHECKLIST.md](DOWNLOAD_CHECKLIST.md)。

## 规范布局

```text
robot_perception_pipeline/data/
├─ models/FastSAM-s.pt
├─ foundationpose/
│  ├─ weights/
│  │  ├─ 2023-10-28-18-33-37/   # refiner + config.yml
│  │  └─ 2024-01-11-20-02-45/   # scorer + config.yml
│  └─ demo_data/mustard0/
└─ datasets/                    # 可选 LINEMOD / YCB
```

FoundationPose **源码**目录在 `config/project.yaml` 的 `foundationpose.repository` 中配置（默认相对路径 `../FoundationPose` 或你本机的克隆位置）。

官方脚本通过 junction 使用本仓库资产（`weights/`、`demo_data/`）。

镜像权重源（脚本默认）：HuggingFace `gpue/foundationpose-weights`  
官方 Google Drive（备用）：https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i  
demo_data Drive：https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP

## FastSAM

```text
robot_perception_pipeline/data/models/FastSAM-s.pt
```

缺失时由 `download_assets.py` 从 CASIA-IVA-Lab/FastSAM releases 拉取。FastSAM 只出掩膜；检测框选 IoU 最大候选后再交给 FoundationPose。

## 公开 6D 数据集

1. FoundationPose `demo_data`：最小冒烟（优先）
2. LINEMOD：较小，适合打通评估
3. YCB-Video：更完整但更大
4. BOP：https://bop.felk.cvut.cz/datasets/

```bash
python run_linemod.py --linemod_dir /path/to/LINEMOD --use_reconstructed_mesh 0
python run_ycb_video.py --ycbv_dir /path/to/YCB_Video --use_reconstructed_mesh 0
```

建议目录：`robot_perception_pipeline/data/datasets/{linemod,ycb_video}/`

## Model-free 选做

参考视图 Drive：https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS  
一周项目先做 model-based（dataset mesh）。

## 相机标定小样例

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_calibration_images.ps1
```

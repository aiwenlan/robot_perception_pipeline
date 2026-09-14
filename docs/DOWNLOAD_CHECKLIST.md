# 资源下载清单（本仓库内路径）

大权重与 demo 数据**自动下载到本仓库** `data/` 下，并已写入 `.gitignore`，不会进 git。

## 一键下载

```powershell
cd <repo-root>
python scripts\download_assets.py
python scripts\link_foundationpose_assets.py
python scripts\check_resources.py
```

脚本会：

1. `pip install -U huggingface_hub gdown`
2. 从 HuggingFace `gpue/foundationpose-weights` 拉取权重到 `data/foundationpose/weights/`
3. 用 `gdown` 拉取官方 `mustard0.zip` / demo_data 到 `data/foundationpose/demo_data/`
4. 若缺失则下载 `data/models/FastSAM-s.pt`
5. 可选：HuggingFace `bop-benchmark/lm` 的 `lm_models.zip` → `data/datasets/linemod/`（仅 mesh，非整段 RGB-D）

**注意：** Google Drive 对 `demo_data` 常出现 *Quota exceeded / Too many users*。权重已有 HF 镜像；demo 若失败脚本会生成 `mustard0_minisynth` 站位场景，并写入 `data/foundationpose/demo_data/README_MANUAL.md`。官方 `mustard0` 仍优先。

## 规范路径

```text
robot_perception_pipeline/
├─ data/models/FastSAM-s.pt
├─ data/foundationpose/weights/
│  ├─ 2023-10-28-18-33-37/model_best.pth   # refiner
│  └─ 2024-01-11-20-02-45/model_best.pth   # scorer
├─ data/foundationpose/demo_data/mustard0/ # rgb/depth/masks/mesh/...
└─ data/datasets/                          # 可选 LINEMOD / YCB（日后）
```

FoundationPose **源码**放在任意本地目录（例如与本仓库同级的 `../FoundationPose`），并在 `config/project.yaml` 的 `foundationpose.repository` 中配置相对或绝对路径。

`link_foundationpose_assets.py` 会在该仓库下创建 Windows **junction**：

- `FoundationPose-main/weights` → 本仓库 `data/foundationpose/weights`
- `FoundationPose-main/demo_data` → 本仓库 `data/foundationpose/demo_data`

这样官方 `python run_demo.py` 无需改路径。

## 检查

```powershell
python scripts\check_resources.py
python scripts\build_demo_manifest.py   # demo_data 就绪后
```

## 可选：公开 6D 小样本

完整 YCB / LINEMOD 很大；需要时放到 `data/datasets/`（已 gitignore）。BOP：https://bop.felk.cvut.cz/datasets/

## Docker 镜像（仍需本机 Docker/WSL）

```bash
docker pull wenbowen123/foundationpose
docker tag wenbowen123/foundationpose foundationpose
```

## 已就绪、无需再下

| 资源 | 路径 |
|---|---|
| OpenCV 棋盘格样例 | `data/calibration/opencv_left/` |
| 合成 RGB-D | `scripts/generate_synthetic_dataset.py` |
| 课程 FP 源码 | `../code/FoundationPose-main` |


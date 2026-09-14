# Windows + WSL2 + Docker 环境配置

## 推荐架构

- Windows 11：文件管理、编辑器和轻量核心测试。
- WSL2 Ubuntu 24.04：ROS2 Jazzy、数据转换和结果评估。
- Docker Desktop（WSL2 backend）：运行 FoundationPose 官方 GPU 镜像。
- NVIDIA Windows 驱动：由 WSL2 和 Docker 共享 GPU；不要在 WSL 内重复安装 Windows 显卡驱动。

参考本机配置：RTX 3060 Laptop 6GB + WSL2 Ubuntu 24.04 + ROS2 Jazzy desktop；Docker Desktop 可选。核心几何 / 标定 / 评估可在 Windows conda 环境运行。

## 1. 安装 WSL2

以管理员身份打开 PowerShell：

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

重启后打开 Ubuntu，创建 Linux 用户，然后：

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git git-lfs build-essential curl python3-venv python3-pip
```

Windows 盘符在 WSL 中映射为 `/mnt/<drive>/...`。可直接从挂载路径运行；大量小文件编译较慢时，建议把 FoundationPose 源码放到 Linux 家目录（如 `~/workspace/FoundationPose`），数据和输出仍可放在 Windows 盘。

## 2. 安装 Docker Desktop 与 GPU 验证

在 Docker Desktop 设置中启用 WSL2 backend，并为 Ubuntu-24.04 开启 WSL Integration。随后在 Ubuntu 中验证：

```bash
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

如果第二条看不到 RTX 3060，优先更新 Windows NVIDIA 驱动、运行 `wsl --update`，再重启 Docker Desktop。

## 3. Windows/WSL 核心 Python 环境

这部分不需要 CUDA：

```bash
cd /path/to/robot_perception_pipeline   # WSL: /mnt/<drive>/.../robot_perception_pipeline
python3 -m venv .venv-core
source .venv-core/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows conda / venv 示例：

```powershell
conda activate <your-env>
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 4. ROS2 Jazzy（WSL Ubuntu 24.04）

按 ROS2 官方 Ubuntu deb 安装页配置软件源并安装 `ros-jazzy-desktop`，再安装项目依赖：

```bash
sudo apt install -y ros-jazzy-desktop ros-jazzy-cv-bridge python3-colcon-common-extensions
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

不要把 ROS2 的系统 Python 包强行装进 FoundationPose 的 Conda 环境。两个环境通过文件或 ROS 消息对接。

官方安装文档：https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

## 5. FoundationPose

权重与 `demo_data` **放在本仓库**（gitignored），再 junction / symlink 到上游 FoundationPose 源码树：

```powershell
cd <repo-root>
python scripts\download_assets.py
python scripts\link_foundationpose_assets.py
```

规范路径：

```text
robot_perception_pipeline/data/foundationpose/weights/.../model_best.pth
robot_perception_pipeline/data/foundationpose/demo_data/mustard0/
```

首选官方 Docker（在上游 FoundationPose 的 `docker/` 目录）：

```bash
cd /path/to/FoundationPose/docker
docker pull wenbowen123/foundationpose
docker tag wenbowen123/foundationpose foundationpose
bash run_container.sh
```

首次进入容器后：

```bash
bash build_all.sh
python run_demo.py
```

6GB 显存若出现 OOM：一次只处理一帧/一个物体，关闭其他 GPU 程序，降低可视化或批量设置；仍失败时先跑官方 demo 的最小帧，再考虑云端 GPU。Conda 原生安装是官方标注的 experimental 方案，不作为首选。

FoundationPose 官方说明：https://github.com/NVlabs/FoundationPose

## 6. 一键检查

```powershell
python scripts\check_environment.py
python scripts\check_resources.py
```

WSL 中把路径分隔符改成 `/` 即可。

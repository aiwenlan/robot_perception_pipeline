#!/usr/bin/env bash
# Install ROS 2 Jazzy (desktop) + project build deps inside Ubuntu 24.04 WSL.
# Usage (from WSL, repo root):
#   bash scripts/wsl_install_ros2_jazzy.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export LANG=en_US.UTF-8

echo "=== locale / base tools ==="
sudo apt-get update -y
sudo apt-get install -y locales curl gnupg lsb-release software-properties-common \
  git git-lfs build-essential python3-pip python3-venv
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "=== ROS 2 apt repo (Tsinghua mirror) ==="
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt-get update -y
sudo mkdir -p /usr/share/keyrings
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
UBU=$(. /etc/os-release && echo "$UBUNTU_CODENAME")
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu ${UBU} main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update -y
sudo apt-get upgrade -y

echo "=== install ros-jazzy-desktop ==="
sudo apt-get install -y ros-jazzy-desktop ros-jazzy-cv-bridge \
  python3-colcon-common-extensions python3-rosdep python3-argcomplete

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init || true
fi
rosdep update || true

BASHRC="$HOME/.bashrc"
MARK='# >>> robot_perception_pipeline ros2 >>>'
if ! grep -Fq "$MARK" "$BASHRC" 2>/dev/null; then
  cat >> "$BASHRC" <<'EOF'

# >>> robot_perception_pipeline ros2 >>>
source /opt/ros/jazzy/setup.bash
# <<< robot_perception_pipeline ros2 <<<
EOF
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u
echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
ros2 --help >/dev/null
echo "=== ROS2 Jazzy install OK ==="

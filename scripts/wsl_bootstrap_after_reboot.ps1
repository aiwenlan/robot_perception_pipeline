# After reboot: finish Ubuntu init + install ROS 2 Jazzy
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_bootstrap_after_reboot.ps1
#
# Optional env:
#   $env:ROS_WSL_USER = "ubuntu"   # default Linux username to create / use

$ErrorActionPreference = "Stop"
$UserName = if ($env:ROS_WSL_USER) { $env:ROS_WSL_USER } else { "ubuntu" }
$ProjectWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectWsl = ($ProjectWin -replace '\\', '/') -replace '^([A-Za-z]):', { '/mnt/' + $args[0].Groups[1].Value.ToLower() }
$RosScript = "$ProjectWsl/scripts/wsl_install_ros2_jazzy.sh"

Write-Host "Project (Windows): $ProjectWin"
Write-Host "Project (WSL):     $ProjectWsl"
Write-Host "WSL user:          $UserName"

Write-Host "=== wsl -l -v ==="
wsl -l -v

$distro = "Ubuntu-24.04"
$listed = (wsl -l -q) -join "`n"
if ($listed -notmatch [regex]::Escape($distro)) {
  if ($listed -match "Ubuntu") {
    $distro = (($listed -split "`n") | Where-Object { $_ -match "Ubuntu" } | Select-Object -First 1).Trim()
  } else {
    Write-Host "No Ubuntu distro yet. Running: wsl --install -d Ubuntu-24.04"
    wsl --install -d Ubuntu-24.04 --no-launch
    Start-Sleep -Seconds 5
  }
}

Write-Host "Using distro: $distro"

Write-Host "=== ensure distro starts ==="
wsl -d $distro -u root -- bash -lc "echo WSL_OK; uname -a; cat /etc/os-release | head -5"

Write-Host "=== create default user '$UserName' if missing (passwordless sudo; set your own password with passwd) ==="
wsl -d $distro -u root -- bash -lc @"
set -e
U='$UserName'
if ! id -u `"\$U`" >/dev/null 2>&1; then
  adduser --disabled-password --gecos '' `"\$U`"
  usermod -aG sudo `"\$U`"
  echo `"\$U ALL=(ALL) NOPASSWD:ALL`" >/etc/sudoers.d/`\$U`
  chmod 440 /etc/sudoers.d/`\$U`
  echo `"Created user \$U. Run: wsl -d $distro -u \$U -- passwd`"
fi
if ! grep -q '^\[user\]' /etc/wsl.conf 2>/dev/null; then
  printf '[user]\ndefault=%s\n' `"\$U`" >> /etc/wsl.conf
fi
"@

wsl --shutdown
Start-Sleep -Seconds 2

Write-Host "=== install ROS 2 Jazzy (long) ==="
wsl -d $distro -u $UserName -- bash -lc "sed -i 's/\r$//' '$RosScript' && bash '$RosScript'"

Write-Host "=== smoke: ros2 ==="
wsl -d $distro -u $UserName -- bash -lc "source /opt/ros/jazzy/setup.bash && ros2 pkg list | head"

Write-Host "DONE. Next: build ros2_ws per docs/ROS2_RUNBOOK.md"

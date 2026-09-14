from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def command_version(command: str, args: list[str]) -> str:
    executable = shutil.which(command)
    if not executable:
        return "missing"
    try:
        output = subprocess.run(
            [executable, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8
        )
        lines = (output.stdout or output.stderr or "no output").splitlines()
        value = lines[0] if lines else "no output"
        console_encoding = sys.stdout.encoding or "utf-8"
        return value.encode(console_encoding, errors="replace").decode(console_encoding)
    except Exception as exc:
        return f"error: {exc}"


print("OS:", platform.platform())
print("Python:", sys.version.split()[0], sys.executable)
for module in ("numpy", "cv2", "scipy", "yaml", "torch", "ultralytics", "rclpy"):
    print(f"{module:12}", "ok" if importlib.util.find_spec(module) else "missing")
print("nvidia-smi: ", command_version("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]))
print("docker:     ", command_version("docker", ["--version"]))
print("wsl:        ", command_version("wsl", ["--version"]))
print("ros2:       ", command_version("ros2", ["--help"]))

import subprocess
import sys
import re

from scripts.musepose_modules.paths import *

def install_package(install_command):
    if "sys_platform" in install_command:
        command = [f"{install_command}"]
    else:
        command = install_command.split()
        print(f"MusePose Extension: Installing {install_command}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )


def get_package_name(line):
    patterns = [
        r'^\s*([a-zA-Z0-9-_]+)\s*==',
        r'^\s*([a-zA-Z0-9-_]+)\s*@',
        r'^\s*([a-zA-Z0-9-_]+)\s*>=',
        r'^\s*([a-zA-Z0-9-_]+)\s*<=',
        r'^\s*([a-zA-Z0-9-_]+)\s*!=',
        r'^\s*([a-zA-Z0-9-_]+)\s*;',
        r'^\s*([a-zA-Z0-9-_]+)\s*$'
    ]

    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            return match.group(1)
    return None

def install_musepose():
    req_file_path = os.path.join(extension_dir, 'musepose_requirements.txt')

    from launch import is_installed, run_pip

    with open(req_file_path) as file:
        for package in file:
            package_name = get_package_name(package)
            if not is_installed(package_name):
                install_package(package)

    problematic_packages = [
        "wheel==0.43.0",
        "mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html",
        "mmengine==0.10.4",
        "mmdet==3.3.0",
        "mmpose==1.3.1",
        "xformers==0.0.23.post1"
    ]

    for command in problematic_packages:
        package_name = get_package_name(command)
        if not is_installed(package_name):
            install_package(command)